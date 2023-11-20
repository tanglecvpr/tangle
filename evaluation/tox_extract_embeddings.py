#----> internal imports
from core.datasets.dataset_mmssl import DatasetFactory
from core.models.mmssl import MMSSL

#----> pytorch imports
import torch
from torch.utils.data import DataLoader

#----> general imports
from tqdm import tqdm
import os
import numpy as np 
import h5py
from collections import OrderedDict


def extract_and_save_embeddings(dataloader, model, savedir):
    model.eval()
    model.cuda()
    with torch.no_grad():
        for idx, (patch_emb, _, patch_positions, image_name, _, _, _) in enumerate(tqdm(dataloader, desc="Extract ssl features ...")):
            patch_emb = patch_emb.cuda()
            patch_positions = patch_positions.cuda()
            wsi_emb = model.get_features(patch_emb, token_position_wsi=patch_positions)
            if not os.path.isdir(savedir):
                os.makedirs(savedir) 
            with h5py.File(os.path.join(savedir, image_name[0]), 'w') as hdf5_file:
                hdf5_file.create_dataset("features", data=np.array(wsi_emb.detach().cpu()))    

def extract_avg_patch_embeddings(dataloader, savedir):
    for idx, (patch_emb, _, patch_positions, image_name, _, _, _) in enumerate(tqdm(dataloader, desc="Extract ssl features ...")):
        if len(patch_emb.shape) != 3:
            raise ValueError("Patch embeddings have wrong dimension")
        patch_emb = patch_emb.mean(dim=1)
        wsi_emb = patch_emb.mean(dim=0)
        if not os.path.isdir(savedir):
            os.makedirs(savedir) 
        with h5py.File(os.path.join(savedir, image_name[0]), 'w') as hdf5_file:
            hdf5_file.create_dataset("features", data=np.array(wsi_emb.detach().cpu())) 
            



# Extract and save ssl slide embeddings

##################### INPUT PARAMS #####################
avg_patch_emebddings = False
extract_all_split_data = False
csv_path = ""
split_csv = ""
patch_feature_dir = ""

ssl_model_paths = [

                    ]

savedir_root = "" 

config = {"wsi_encoder": "abmil",
          "rna_encoder": "mlp",
          "intra_modality_rna": False,
          "patch_embedding_dim": 768,
          "embedding_dim": 768,
          "activation": "softmax",
          "tokens_rna": 1000}

########################################################

for idx, path in enumerate(ssl_model_paths):
    if path is not None:
        if not os.path.exists(path):
            raise ValueError(f"Invalid path in ssl_model_paths at position {idx}")
    
if avg_patch_emebddings:
    print("WARNING: Extract averaged patch embeddings")

for i, ssl_model_path in enumerate(tqdm(ssl_model_paths, desc=f"Extract embeddings for different models")):

    savedir = savedir_root + ssl_model_path.split("/")[-3] if not avg_patch_emebddings else savedir_root + "avg_patch_embeddings"

    # create dataset and inference SSL model
    dataset_factory = DatasetFactory(patch_feature_dir=patch_feature_dir,
                                            csv_path=csv_path,
                                            rnaseq_path=None,
                                            split_file_path=split_csv,
                                            n_tokens=-1,
                                            sampling_strategy=None,
                                            sampling_augmentation=None,
                                            prune_compunds_ssl=False,
                                            prune_compunds_downstream=False,
                                            normalization_mode=None,
                                            prune_genes_1k=False,
                                            )
    
    if extract_all_split_data:
        split_dataset = dataset_factory.return_all()
        split_dataloader = DataLoader(dataset=split_dataset, batch_size=1)
        savedir_split = savedir + "/all_features"
    else:
        val_split = dataset_factory.return_val_split()
        val_split_dataloader = DataLoader(dataset=val_split, batch_size=1)
        test_split = dataset_factory.return_test_split()
        test_split_dataloader = DataLoader(dataset=test_split, batch_size=1)
        curated_test_split = dataset_factory.return_cur_test_split()
        curated_test_split_dataloader = DataLoader(dataset=curated_test_split, batch_size=1)
        savedir_val_split = savedir + "/val_split" 
        savedir_test_split = savedir + "/test_split" 
        savedir_curated_test_split = savedir + "/curated_test_split"
    
    if ssl_model_path is None and avg_patch_emebddings:
        if extract_all_split_data:
            extract_avg_patch_embeddings(dataloader=split_dataloader,
                                     savedir=savedir_split)
        else:
            extract_avg_patch_embeddings(dataloader=val_split_dataloader,
                                        savedir=savedir_val_split)
            extract_avg_patch_embeddings(dataloader=test_split_dataloader,
                                        savedir=savedir_test_split)
            extract_avg_patch_embeddings(dataloader=curated_test_split_dataloader,
                                        savedir=savedir_curated_test_split)
        
    else:   
        ssl_model = MMSSL(config=config, 
                        n_tokens_wsi=None, 
                        n_tokens_rna=config["tokens_rna"],
                        patch_embedding_dim=config["patch_embedding_dim"])
        
        state_dict = torch.load(ssl_model_path)
        # deal with key names when pretrained with DataParallel wrapper
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.' prefix
            else:
                name = k
            new_state_dict[name] = v
        ssl_model.load_state_dict(state_dict=new_state_dict)

        # Do inference and save embeddings to file
        if extract_all_split_data:
            extract_and_save_embeddings(dataloader=split_dataloader,
                                        model=ssl_model,
                                        savedir=savedir_split)
        else:
            extract_and_save_embeddings(dataloader=val_split_dataloader, 
                                                            model=ssl_model, 
                                                            savedir=savedir_val_split)
            extract_and_save_embeddings(dataloader=test_split_dataloader, 
                                                            model=ssl_model, 
                                                            savedir=savedir_test_split)                                 
            extract_and_save_embeddings(dataloader=curated_test_split_dataloader, 
                                                            model=ssl_model, 
                                                            savedir=savedir_curated_test_split
                                                            )


