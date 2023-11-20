#----> internal imports
from core.datasets.dataset_mmssl import DatasetFactory
from core.trainer_mmssl import MMSSLTrainer

#----> pytorch imports
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

#----> general imports 
import os


class ExperimentHandler:

    def __init__(self, config):
        self.Trainer = None
        self.model = None
        self.config = config
        self.tb_log = SummaryWriter(log_dir=os.path.join(self.config["results_dir"], "tb"))

    def run_experiment(self):

        if self.config["intra_modality_mode_wsi"] == "contrast_token_views" or self.config["intra_modality_mode_wsi"] == "contrast_token_views+contrast_avg_emb":
            sampling_augmentation = True
        else:
            sampling_augmentation = False
            
        dataset_factory = DatasetFactory(patch_feature_dir=self.config["patch_feature_dir"],
                                         csv_path=self.config["csv_path"],
                                         rnaseq_path=self.config["rnaseq_path"],
                                         split_file_path=self.config["split_file_path"],
                                         n_tokens=self.config["n_tokens"],
                                         sampling_strategy=self.config["sampling_strategy"],
                                         sampling_augmentation=sampling_augmentation,
                                         prune_compunds_ssl=self.config["prune_compunds_ssl"],
                                         prune_compunds_downstream=self.config["prune_compunds_downstream"],
                                         normalization_mode=self.config['rna_normalization'],
                                         prune_genes_1k=self.config['prune_genes_1k'],
                                        )
        
        train_split, val_split, test_split, curated_test_split = dataset_factory.return_splits()
        
        dataset_summary = f'\n ####### Datasplit summary ########\nsamples train split: {train_split.__len__()}\nsamples val split: {val_split.__len__()}\nsamples test split: {test_split.__len__()}\nsamples curated test split: {curated_test_split.__len__()}\n'
        print(dataset_summary)
        self.tb_log.add_text('num_data_samples', dataset_summary)

        self.Trainer = MMSSLTrainer(config=self.config)

        results = self.Trainer.train_val_test(train_split=train_split, 
                                              val_split=val_split, 
                                              test_split=test_split,
                                              curated_test_split=curated_test_split,
                                              config=self.config)
        
        results = pd.DataFrame([results])
        results.insert(0, "exp_name", self.config["exp_name"])
        results.to_csv(os.path.join(self.config["results_dir"], "results.csv"), index=False)
    
        return results