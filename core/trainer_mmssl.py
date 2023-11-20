#----> internal imports
from core.utils.info_nce_loss import InfoNCE
from core.datasets.dataset_mmssl import SlideEmbeddingDataset
from core.utils.utils_mmssl import save_roc, EarlyStopping, print_network, smooth_rank_measure
from core.models.mmssl import MMSSL


#----> pytorch imports
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter

#----> general imports 
import os
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import shutil
import time
from PIL import Image
from tqdm import tqdm

class MMSSLTrainer():

    def __init__(self, config):
        self.config = config
        self.ssl_model = None
        self.optimizer = None
        self.inter_modality_loss = None
        self.intra_rna_loss = None
        self.intra_wsi_loss = None
        self.scheduler_warmup = None
        self.scheduler = None
        self.writer = SummaryWriter(log_dir=os.path.join(self.config['results_dir'], 'tb'))
        self.results = {"test_bin":None, 
                        "test_avgemb_bin":None,
                        "test_mult":None,
                        "test_avgemb_mult":None,
                        "val_bin":None,
                        "val_avgemb_bin": None,
                        "val_mult":None,
                        "val_avgemb_mult":None,
                        "test_mult_class":None,
                        "test_avgemb_mult_class":None,
                        "val_mult_class":None,
                        "val_avgemb_mult_class":None,
                        "class_names_test":None,
                        "class_names_val":None,  
                        }
        if 'debug' in self.config.keys():
            if self.config['debug']:
                self.debug_mode_on = True
            else:
                self.debug_mode_on = False
        else:
            self.debug_mode_on = False

    ######################################################################################################### 
    #                                                                                                       #
    #                                  CORE FUNCTION AND TRAINING SETUP                                     #
    #                                                                                                       #
    ######################################################################################################### 
    
    def train_val_test(self, train_split, val_split, test_split, curated_test_split, config):
        
        #---> log experiment details
        self.writer.add_text("config file", str(self.config)) 

        #---> init loss functions
        if self.config["intra_modality_wsi"]:
            if self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb" or  self.config["intra_modality_mode_wsi"] == "contrast_token_views+contrast_avg_emb":
                self.intra_wsi_loss = self.init_intra_wsi_loss_function()
                self.intra_wsi_loss_2 = self.init_intra_wsi_loss_function()
            self.intra_wsi_loss = self.init_intra_wsi_loss_function()
        if self.config["intra_modality_rna"]:
            self.intra_rna_loss = self.init_intra_rna_loss_function()
        if self.config["inter_modality"]:  
            if self.config["inter_modality_pathOmics"]:
                self.inter_modality_loss = nn.MSELoss(reduction=self.config["mse_reduction"])
            else:
                self.inter_modality_loss = self.init_inter_modality_loss_function()
        
        #---> init dataloaders
        train_loader, val_loader, test_loader, curated_testloader = self.init_loaders(config=config, 
                                                                                        train_split=train_split, 
                                                                                        val_split=val_split, 
                                                                                        test_split=test_split,
                                                                                        curated_test_split=curated_test_split
                                                                                        )
        
        # --> eval averaged patch embedding features before training
        if self.config['eval_patch_features']:
            print('\nStart evaluation before trainig (average patch features) ... ')
            self.validate_patch_embeddings(train_dataloader=val_loader, val_dataloader=test_loader, test_dataloder=curated_testloader,ep=0)
        
        # --> load ssl load model
        self.ssl_model = MMSSL(config=self.config, 
                               n_tokens_wsi=self.config["n_tokens"], 
                               n_tokens_rna=train_split.__getitem__(0)[1].shape[0],
                               patch_embedding_dim=train_split.__getitem__(0)[0].shape[1])
        print_network(net=self.ssl_model, results_dir=self.config['results_dir'])
        if len(self.config['gpu_devices']) > 1:
            self.ssl_model = nn.DataParallel(self.ssl_model, device_ids=self.config["gpu_devices"])
            self.ssl_model.to(f'cuda:{self.config["gpu_devices"][0]}')
        else:
            self.ssl_model.cuda() if torch.cuda.is_available() else "cpu"
            
        #---> init optimizer
        self.optimizer = self.get_optimizer(opt=self.config["optimizer"], 
                                            lr=self.config['learning_rate'],
                                            model=self.ssl_model)
        
        #---> init scheduler        
        T_max = (self.config["epochs"] - self.config["warmup_epochs"]) * len(train_loader) if self.config["warmup"] else self.config["epochs"] * len(train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, 
                                           T_max = T_max,
                                           eta_min = self.config["end_learning_rate"]
                                           )
        if self.config["warmup"]:
            self.scheduler_warmup = LinearLR(self.optimizer, 
                                            start_factor=0.00001,
                                            total_iters=self.config["warmup_epochs"] * len(train_loader))
        else:
            self.scheduler_warmup = None   
            
        # --> evaluate random features from ssl model
        if self.config['eval_random_init_model']:
            print('\nStart evaluation before trainig (random init model) ... ')
            _, _ = self.validate(train_dataloader=val_loader, ep=0, val_dataloader=test_loader)
            
        # ---> train val ssl model
        best_downstream_model_binary, best_downstream_model_multilabel = self.step(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    
        #---> test on curated testset
        s_test_time = time.time()
        auc_bin, _ = self.test(test_dataloader=curated_testloader, model=best_downstream_model_binary, mode="binary")
        auc_mult, auc_per_class = self.test(test_dataloader=curated_testloader, model=best_downstream_model_multilabel, mode="multi")
        e_test_time = time.time()
        test_time = np.round((e_test_time- s_test_time)/60, 2)
        self.writer.add_scalar('time & util linear validation/test time [min]', test_time, global_step=self.config["epochs"])
        self.results.update({key: value for key, value in zip(["test_bin", "test_mult", "test_mult_class"], [auc_bin, auc_mult, str(auc_per_class)])})

        return self.results
            
    
    ######################################################################################################### 
    #                                                                                                       #
    #                                  MULTI-MODAL SSL PRE-TRAINING                                         #
    #                                                                                                       #
    ######################################################################################################### 
    
    def step(self, train_loader, val_loader, test_loader):
        
        # logging
        self.writer.add_text('patch_feature_dir', str(self.config['patch_feature_dir']))
        self.writer.add_text('experiment name', str(self.config['exp_name']))
            
        # train loop
        print('\nStart Training...')
        for epoch in range(1,self.config['epochs']+1):
            s_ep = time.time()
            self.train_loop(epoch=epoch, dataloader=train_loader)
            
            # save model to checkpoint
            if not os.path.isdir(os.path.join(self.config["results_dir"], "checkpoints_ssl")):
                    os.mkdir(os.path.join(self.config["results_dir"], "checkpoints_ssl"))
            name = "checkpoint{}.pth".format(epoch) if epoch % self.config["save_interval"] == 0 else "checkpoint_last.pth"
            torch.save(self.ssl_model.state_dict(), os.path.join(self.config["results_dir"], "checkpoints_ssl", name))
            e_ep = time.time()
            ep_time = np.round((e_ep-s_ep)/60, 2)
            self.writer.add_scalar("time & util ssl training/total epoch time [min]", ep_time, global_step=epoch)
            
            # validate embeddings on downstream task - training linear classifier with early stopping approach 
            s_val = time.time()
            if epoch % self.config["val_interval_ep"] == 0 or epoch == self.config["epochs"]:
                downstream_model_binary, downstream_model_multilabel = self.validate(train_dataloader=val_loader, ep=epoch, val_dataloader=test_loader)
            e_val = time.time()
            val_time = np.round((e_val-s_val)/60, 2)
            self.writer.add_scalar('time & util linear validation/total val time [min]', val_time, global_step=epoch)
        
        return downstream_model_binary, downstream_model_multilabel

    
    def train_loop(self, epoch, dataloader):
        
        self.ssl_model.train()

        ep_loss = 0.
        ep_recon_loss = 0.
        ep_inter_loss = 0.
        ep_intra_wsi_loss = 0.
        fb_time = 0.
        print('\n')
        for b_idx, (patch_emb, rna_seq, patch_positions, _, patch_emb_aug, patch_positions_aug, avg_patch_emb) in enumerate(dataloader):
            
            losses = []    
            
            if self.config["intra_modality_wsi"] and not self.config["inter_modality"] and not self.config["intra_modality_rna"]:
                rna_seq = None
            if self.config["intra_modality_wsi"]:
                if self.config["intra_modality_mode_wsi"] == "contrast_token_views" or self.config["intra_modality_mode_wsi"]== "contrast_token_views+contrast_avg_emb":
                    patch_emb = torch.cat((patch_emb, patch_emb_aug))
                    patch_positions = torch.cat((patch_positions, patch_positions_aug))
                elif self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb" or self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb" or self.config["intra_modality_mode_wsi"] == "contrast_token_views+contrast_avg_emb":
                    patch_emb_mask, patch_positions_mask = self.apply_random_mask(patch_embeddings=patch_emb, patch_positions=patch_positions)
                    patch_emb = torch.cat((patch_emb, patch_emb_mask))
                    patch_positions = torch.cat((patch_positions, patch_positions_mask))
                    
            # set data on device 
            patch_emb = patch_emb.cuda()
            rna_seq = rna_seq.cuda() if rna_seq is not None else rna_seq
            # patch positions are normalized for x and y
            patch_positions = patch_positions.cuda() 
            if self.config["intra_modality_mode_wsi"] == "contrast_avg_emb" or self.config["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb" or self.config["intra_modality_mode_wsi"] == "contrast_token_views+contrast_avg_emb":
                avg_patch_emb = avg_patch_emb.cuda()

            s_fb = time.time()
            
            # forward pass and loss 
            if self.config["intra_modality_wsi"] and not self.config["inter_modality"] and not self.config["intra_modality_rna"]:
                wsi_emb, _, _ = self.ssl_model(patch_emb, None, token_position_wsi=patch_positions)
            else:
                wsi_emb, rna_emb, rna_reconstruction = self.ssl_model(patch_emb, rna_seq, token_position_wsi=patch_positions)
            
            # inter modality loss wsi <-> rna
            if self.config["inter_modality"]:
                if self.config["inter_modality_pathOmics"]:
                    losses.append(self.inter_modality_loss(wsi_emb, rna_emb))
                else:
                    if self.config["intra_modality_mode_wsi"] == "contrast_token_views" or self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
                        split_idx = int(patch_emb.shape[0]/2)
                        losses.append(self.inter_modality_loss(query=wsi_emb[:split_idx], positive_key=rna_emb, symmetric=self.config["symmetric_cl"]))
                    else:
                        losses.append(self.inter_modality_loss(query=wsi_emb, positive_key=rna_emb, symmetric=self.config["symmetric_cl"]))
                    ep_inter_loss += losses[-1].item()
            
            # intra modality loss wsi <-> wsi
            if self.config["intra_modality_wsi"]:
                if self.config["intra_modality_mode_wsi"] == "contrast_token_views" or self.config["intra_modality_mode_wsi"] == "contrast_token_views+contrast_avg_emb":
                    split_idx = int(patch_emb.shape[0]/2)
                    l = self.intra_wsi_loss(query=wsi_emb[:split_idx], positive_key=wsi_emb[split_idx:], symmetric=self.config["symmetric_cl"])
                    if self.config["contrast_token_views_loss_scale"]:
                        losses.append(l*self.config["embedding_dim"])
                    else:
                        losses.append(l)
                    if self.config["intra_modality_mode_wsi"] == "contrast_token_views+contrast_avg_emb":
                        losses.append(self.intra_wsi_loss_2(query=wsi_emb[:split_idx], positive_key=avg_patch_emb, symmetric=self.config["symmetric_cl"]))
                elif self.config["intra_modality_mode_wsi"] == "contrast_avg_emb":
                    losses.append(self.intra_wsi_loss(query=wsi_emb, positive_key=avg_patch_emb, symmetric=self.config["symmetric_cl"]))
                elif self.config["intra_modality_mode_wsi"] == "reconstruct_avg_emb":
                    losses.append(self.intra_wsi_loss(wsi_emb, avg_patch_emb))
                elif self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
                    split_idx = int(patch_emb.shape[0]/2)
                    losses.append(self.intra_wsi_loss(wsi_emb[split_idx:], wsi_emb[:split_idx])) # 1. masked wsi_emb 2. umasked wsi_emb
                elif self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
                    split_idx = int(patch_emb.shape[0]/2)
                    losses.append(self.intra_wsi_loss(wsi_emb[split_idx:], wsi_emb[:split_idx])) # 1. masked wsi_emb 2. umasked wsi_emb
                    losses.append(self.intra_wsi_loss(query=wsi_emb[:split_idx], positive_key=avg_patch_emb, symmetric=self.config["symmetric_cl"]))
                else:
                    raise ValueError("Invalid intra_modality_mode_wsi.")
                ep_intra_wsi_loss += losses[-1].item()
                
            # intra modality loss rna <-> rna
            if self.config["intra_modality_rna"]:
                losses.append(self.intra_rna_loss(rna_reconstruction, rna_seq))
                ep_recon_loss += losses[-1].item()
                
            loss = sum(losses)
            
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            
            e_fb = time.time()
            fb_time += e_fb - s_fb

            if epoch <= self.config["warmup_epochs"]:
                self.scheduler_warmup.step()
            else:
                self.scheduler.step()            

            ep_loss += loss.item()

            # print log
            if b_idx == 0:
                print(f"free memory [gb]: {torch.cuda.mem_get_info()[0] * 1.0E-09}")
            lr = self.scheduler_warmup.get_last_lr()[0] if epoch <= self.config["warmup_epochs"] else self.scheduler.get_last_lr()[0]
            self.writer.add_scalar("params ssl/lr", lr, global_step=(epoch-1)*len(dataloader)+b_idx)
            self.writer.add_scalar("params ssl/batch_size", dataloader.batch_size, global_step=(epoch-1)*len(dataloader)+b_idx)
            mem_free = {f'cuda:{i}': torch.cuda.mem_get_info(device=i)[0] * 1.0E-09 for i, num in enumerate(range(torch.cuda.device_count()))}
            mem_tot = {f'cuda:{i}': torch.cuda.mem_get_info(device=i)[1] * 1.0E-09 for i, num in enumerate(range(torch.cuda.device_count()))}
            self.writer.add_scalars("time & util ssl training/free memory [gb]", mem_free, global_step=(epoch-1)*len(dataloader)+b_idx)
            self.writer.add_scalars("time & util ssl training/total memory [gb]", mem_tot, global_step=(epoch-1)*len(dataloader)+b_idx)
            self.writer.add_scalar("params ssl/tokens", patch_emb.shape[1], global_step=(epoch-1)*len(dataloader)+b_idx)
            if b_idx > 0 and b_idx % self.config['log_interval'] == 0:
                print(f'## Epoch {epoch}, Step {b_idx}/{len(dataloader)} - Loss: {ep_loss/b_idx}, lr: {lr}')
        
        # print and log
        fb_time = np.round((fb_time)/60, 2)
        ep_loss /= len(dataloader)
        ep_inter_loss /= len(dataloader)
        ep_intra_wsi_loss /= len(dataloader)
        ep_recon_loss /= len(dataloader)
        self.writer.add_scalar("params ssl/loss_ep", ep_loss, global_step=epoch)
        self.writer.add_scalar("params ssl/loss inter modality", ep_inter_loss, global_step=epoch)
        self.writer.add_scalar("params ssl/loss intra modality wsi", ep_intra_wsi_loss, global_step=epoch)
        self.writer.add_scalar("params ssl/loss rna reconstruction", ep_recon_loss, global_step=epoch)
        self.writer.add_scalar("time & util ssl training/f/b epoch time [min]", fb_time, global_step=epoch)
        print (f'\n## Final Loss epoch {epoch}: {ep_loss}')


    ######################################################################################################### 
    #                                                                                                       #
    #                                  DOWNSTREAM EVALUATION AND TEST                                       #
    #                                                                                                       #
    ######################################################################################################### 

    def validate(self, train_dataloader, ep, val_dataloader=None):
        
        print("\n----------- Validation, train linear classifier... -----------")
        
        # extract features from SSL pretraining for linear classifier training + validation (early stopping)
        s_emb = time.time()
        feature_folder_path_train, rank_val = self.extract_and_save_embeddings(dataloader=train_dataloader, 
                                                                    model=self.ssl_model, 
                                                                    savedir=os.path.join(self.config["results_dir"], "train_temp"),
                                                                    eval_embedding_rank=True)
        feature_folder_path_val, _ = self.extract_and_save_embeddings(dataloader=val_dataloader, 
                                                                        model=self.ssl_model,
                                                                        savedir=os.path.join(self.config["results_dir"], "val_temp"))
        e_emb = time.time()
        time_emb = np.round((e_emb- s_emb)/60, 2)
        self.writer.add_scalar('time & util linear validation/feature extract. time [min]', time_emb, global_step=ep+1)

        # create SSL feature datasets 
        csv = pd.read_csv(self.config["csv_path"])
        feature_train_dataset = SlideEmbeddingDataset(feature_folder_path=feature_folder_path_train, 
                                                      dataset_csv=csv, 
                                                      lesions=self.config["lesions"], 
                                                      equal_normal_abnormal=self.config["equal_normal_abnormal"],
                                                      do_testset_label_modifications=self.config["do_testset_label_modifications"])
        feature_train_dataloader = DataLoader(dataset=feature_train_dataset, batch_size=self.config["ds_batch_size"])
        feature_val_dataset = SlideEmbeddingDataset(feature_folder_path=feature_folder_path_val, 
                                                    dataset_csv=csv, 
                                                    lesions=self.config["lesions"], 
                                                    equal_normal_abnormal=self.config["equal_normal_abnormal"],
                                                    do_testset_label_modifications=self.config["do_testset_label_modifications"])
        feature_val_dataloader = DataLoader(dataset=feature_val_dataset, batch_size=self.config["ds_inf_batch_size"])
        self.writer.add_text('linear classifier train dataset sizes', str(f'train_lin: {feature_train_dataset.__len__()}, val_lin: {feature_val_dataset.__len__()}'))
        
        # train and evaluate linear classifiers
        s_val_t = time.time()
        downstream_model_binary, best_auc_binary, fb_time_bin, _ = self.train_linear_classifier(train_dataloader=feature_train_dataloader, mode="binary", val_dataloader=feature_val_dataloader)
        downstream_model_multilabel, best_auc_multi, fb_time_mult, best_auc_multi_per_class = self.train_linear_classifier(train_dataloader=feature_train_dataloader, mode="multi", val_dataloader=feature_val_dataloader)
        e_val_t = time.time()
        time_val_t = np.round((e_val_t- s_val_t)/60, 2)
        time_fb_val = np.round((fb_time_bin+fb_time_mult)/60, 2)
        
        # logging
        self.writer.add_scalar('time & util linear validation/train linear time [min]', time_val_t, global_step=ep)
        self.writer.add_scalar('time & util linear validation/train linear fb time [min]', time_fb_val, global_step=ep)
        self.writer.add_scalar('results validation/auc multilabel', best_auc_multi, global_step=ep)
        self.writer.add_scalar('results validation/auc binary', best_auc_binary, global_step=ep)
        self.writer.add_scalar('results validation/embedding rank', rank_val, global_step=ep)
        per_class_data = dict(zip(feature_val_dataset.lesions, best_auc_multi_per_class))
        self.writer.add_scalars('results validation/auc multilabel per class', per_class_data, global_step=ep)
        self.results[f"val_bin"] = best_auc_binary
        self.results[f"val_mult"] = best_auc_multi
        self.results[f"val_mult_class"] = best_auc_multi_per_class
        
        # remove feature folder if needed
        if feature_folder_path_train.endswith("temp") and os.path.isdir(feature_folder_path_train):
            shutil.rmtree(feature_folder_path_train)
        if feature_folder_path_val.endswith("temp") and os.path.isdir(feature_folder_path_val):
            shutil.rmtree(feature_folder_path_val)
            
        return downstream_model_binary, downstream_model_multilabel 

    def validate_patch_embeddings(self, train_dataloader, val_dataloader, test_dataloder, ep):
        
        print("\n----------- Validation Patch Embeddings, train linear classifier... -----------")

        # create SSL feature datasets 
        csv = pd.read_csv(self.config["csv_path"])
        feature_train_dataset = SlideEmbeddingDataset(feature_folder_path=self.config['patch_feature_dir'], 
                                                    ids=train_dataloader.dataset.ids, 
                                                    dataset_csv=csv, 
                                                    lesions=self.config["lesions"],
                                                    equal_normal_abnormal=self.config["equal_normal_abnormal"],
                                                    do_testset_label_modifications=self.config["do_testset_label_modifications"])
        feature_train_dataloader = DataLoader(dataset=feature_train_dataset, batch_size=self.config["ds_batch_size"])
        
        feature_val_dataset = SlideEmbeddingDataset(feature_folder_path=self.config['patch_feature_dir'], 
                                                  ids=val_dataloader.dataset.ids, 
                                                  dataset_csv=csv, 
                                                  lesions=self.config["lesions"],
                                                  equal_normal_abnormal=self.config["equal_normal_abnormal"],
                                                  do_testset_label_modifications=self.config["do_testset_label_modifications"])
        feature_val_dataloader = DataLoader(dataset=feature_val_dataset, batch_size=self.config["ds_inf_batch_size"])
        
        # logging
        self.writer.add_text('linear classifier train dataset sizes', str(f'train_lin: {feature_train_dataset.__len__()}, val_lin: {feature_val_dataset.__len__()}'))
        summary = '######## TRAIN ########\n'
        train_summary = feature_train_dataset.print_dataset_summary()
        summary += train_summary
        summary += '######## VAL ########\n'
        validation_summary = feature_val_dataset.print_dataset_summary()
        summary += validation_summary
        
        # train and evaluate linear classifiers
        downstream_model_binary, best_auc_binary, _, _ = self.train_linear_classifier(train_dataloader=feature_train_dataloader, mode="binary", val_dataloader=feature_val_dataloader, patch_features=True)
        downstream_model_multilabel, best_auc_multi, _, auc_mult_per_class_val = self.train_linear_classifier(train_dataloader=feature_train_dataloader, mode="multi", val_dataloader=feature_val_dataloader, patch_features=True)

        if test_dataloder:
            feature_test_dataset = SlideEmbeddingDataset(feature_folder_path=self.config['patch_feature_dir'], 
                                                    ids=test_dataloder.dataset.ids, 
                                                    dataset_csv=csv, 
                                                    lesions=self.config["lesions"],
                                                    equal_normal_abnormal=False,
                                                    do_testset_label_modifications=self.config["do_testset_label_modifications"])
            feature_test_dataloader = DataLoader(dataset=feature_test_dataset, batch_size=self.config["ds_inf_batch_size"])
            
            summary += '######## TEST ########\n'
            test_summary = feature_test_dataset.print_dataset_summary()
            summary += test_summary
            
            auc_bin, _ = self.test_linear_classifier(dataloader=feature_test_dataloader, model=downstream_model_binary, mode="binary")
            auc_mult, auc_mult_per_class_test = self.test_linear_classifier(dataloader=feature_test_dataloader, model=downstream_model_multilabel, mode="multi")
            
            # log
            self.results["test_avgemb_bin"] = auc_bin
            self.results["test_avgemb_mult"] = auc_mult
            self.results["test_avgemb_mult_class"] = str(auc_mult_per_class_test)
            self.results["class_names_test"] = str(feature_test_dataset.lesions)
            self.writer.add_scalar(f'results test/avg_emb testset binary', auc_bin, global_step=self.config["epochs"])
            self.writer.add_scalar(f'results test/avg_emb testset multi', auc_mult, global_step=self.config["epochs"])
            per_class_data_test = dict(zip(feature_test_dataset.lesions, auc_mult_per_class_test))
            self.writer.add_scalars('results test/avg_emb testset multi per class', per_class_data_test, global_step=ep)
            
            
            
        # log
        print(summary)
        with open(os.path.join(self.config["results_dir"],'downstream_dataset_summary.txt'), 'w') as f:
            f.write(summary)
        self.writer.add_scalar('results validation/patch emb. auc multilabel', best_auc_multi, global_step=ep)
        per_class_data_val = dict(zip(feature_val_dataset.lesions, auc_mult_per_class_val))
        self.writer.add_scalars('results validation/patch emb. auc multilabel per class', per_class_data_val, global_step=ep)
        self.writer.add_scalar('results validation/patch emb auc binary', best_auc_binary, global_step=ep)
        self.results["val_avgemb_bin"] = best_auc_binary
        self.results["val_avgemb_mult"] = best_auc_multi
        self.results["class_names_val"] = str(feature_val_dataset.lesions)
        self.results["val_avgemb_mult_class"] = str(auc_mult_per_class_val)
    
        
    def test(self, test_dataloader, model, mode):
        
        print(f"\n---------- Test {mode} model ... -----------")
        
        # create SSL feature datasets 
        csv = pd.read_csv(self.config["csv_path"])
        savepath = os.path.join(self.config["results_dir"], "ssl_testset_features") if self.config["save_testset_features"] else os.path.join(self.config["results_dir"], "test_temp")
        path, _ = self.extract_and_save_embeddings(dataloader=test_dataloader, 
                                                model=self.ssl_model, 
                                                savedir=savepath)
        feature_dataset = SlideEmbeddingDataset(feature_folder_path=path, 
                                                dataset_csv=csv, 
                                                lesions=self.config["lesions"], 
                                                equal_normal_abnormal=False,
                                                do_testset_label_modifications=self.config["do_testset_label_modifications"])
        feature_dataloader = DataLoader(dataset=feature_dataset, batch_size=self.config["ds_inf_batch_size"])
        self.writer.add_text('linear classifier test dataset sizes', str(f'test_lin: {feature_dataset.__len__()}'))
        
        # sanity check
        out_dim = 1 if mode=="binary" else len(feature_dataloader.dataset.lesions)
        check_model = self.init_downstream_model(out_dim=out_dim, in_dim=self.config['embedding_dim'])
        check_dic = torch.load(os.path.join(self.config["results_dir"], f"downstream_model_{mode}"))
        check_model.load_state_dict(check_dic)
        if torch.any(model.weight[0] != check_model.weight[0]):
            raise Warning("Loaded Test model unequals saved test model.")
    
        # testset inference and metric measurement
        auc, auc_per_class = self.test_linear_classifier(dataloader=feature_dataloader, model=model, mode=mode, save_roc_curves=True)
        
        # logging
        self.writer.add_scalar(f'results test/macro auc curated testset {mode}', auc, global_step=self.config["epochs"])
        if mode != "binary":
            per_class_data_test = dict(zip(feature_dataset.lesions, auc_per_class))
            self.writer.add_scalars('results test/macro auc curated testset multi per class', per_class_data_test, global_step=self.config["epochs"])
            
        # remove feature folder if needed
        if path.endswith("temp") and os.path.isdir(path):
            shutil.rmtree(path)
        
        return auc, auc_per_class
    
    
    def train_linear_classifier(self, train_dataloader, mode, val_dataloader, patch_features=False):
        
        in_dim = train_dataloader.dataset.__getitem__(0)[0].shape[0] if patch_features else self.config['embedding_dim']
        out_dim = 1 if mode=="binary" else len(train_dataloader.dataset.lesions)
        savepath = os.path.join(self.config["results_dir"], f"downstream_model_{mode}")
        
        bcewlogits = torch.nn.BCEWithLogitsLoss()
    
        # train loop over learning rates
        lrs = self.config["ds_learning_rates"]
        lr_losses = []
        lr_scores = []
        lr_scores_train = []
        lr_scores_per_class = []
        best_lr_models = []
        best_lr_scores = []
        best_lr_scores_per_class = []
        fb_time = 0.0
        for idxx, lr in enumerate(tqdm(lrs, desc=f'Train lin. classifier, mode: {mode}') if val_dataloader else print(f'Train linear classifiers')):
        
            model = self.init_downstream_model(in_dim=in_dim, out_dim=out_dim)
            
            optimizer = torch.optim.AdamW(lr=lr, params=model.parameters())
            
            if self.config['early_stopping']:
                es = EarlyStopping(patience=15, stop_epoch=5)

            ep_losses = []
            ep_scores = []
            ep_scores_train = []
            ep_scores_per_class = []
            for ep in range(self.config["ds_epochs"]):
                
                model.train()
                
                ep_loss = 0.
                for features, class_binary, class_multi, class_names, img_name  in train_dataloader:
                    
                    features = features.cuda() if torch.cuda.is_available() else features
                    labels = class_binary if mode == 'binary' else class_multi
                    labels = labels.cuda() if torch.cuda.is_available() else labels
    
                    s_fb = time.time()
                    
                    out_raw = model(features)

                    loss = bcewlogits(out_raw, labels) 
                    ep_loss += loss.detach().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    e_fb = time.time()
                    fb_time += (e_fb-s_fb)
                    
                ep_loss /= len(train_dataloader)
                ep_losses.append(ep_loss)
                
                model.eval()
                
                # train and validation scores
                ep_score_train, _ = self.test_linear_classifier(model=model, dataloader=train_dataloader, mode=mode)
                ep_scores_train.append(ep_score_train)
                ep_score, ep_score_per_class = self.test_linear_classifier(model=model, dataloader=val_dataloader, mode=mode)
                ep_scores.append(ep_score)  
                ep_scores_per_class.append(ep_score_per_class)
                
                # save best or last model 
                if self.config['early_stopping']:
                    es(epoch=ep, score=ep_score, model=model)
                    lr_best_model = es.best_model
                    if es.early_stop:
                        break
                else:
                    lr_best_model = model
                            
            # save best model and score for one run
            best_lr_models.append(lr_best_model)
            if self.config['early_stopping']:
                best_lr_scores.append(es.best_score)
                idx_best_score = np.where(ep_scores==es.best_score)[0][0].item()
                best_lr_scores_per_class.append(ep_scores_per_class[idx_best_score])
            else:
                best_lr_scores.append(ep_score)
                best_lr_scores_per_class.append(ep_score_per_class)
            
            
            # for logging
            lr_scores.append(ep_scores)
            lr_scores_train.append(ep_scores_train)
            lr_losses.append(ep_losses)
        
        lr_idx = np.argmax(best_lr_scores)
        best_model = best_lr_models[lr_idx]

        torch.save(best_model.state_dict(), savepath)
        
        # logging metrics 
        for i, tlrl in enumerate(lr_losses):
            for tep, lo in enumerate(tlrl):
                self.writer.add_scalars(f'results validation/train loss {mode} - last validation', {'lr: {}'.format(lrs[i]): lo}, tep)
        for k, tlrs in enumerate(lr_scores):
            for trep, sc in enumerate(tlrs):
                self.writer.add_scalars(f'results validation/val auc {mode} - last validation', {'lr: {}'.format(lrs[k]): sc}, trep)   
        for k, tlrs in enumerate(lr_scores_train):
            for trep, sc in enumerate(tlrs):
                self.writer.add_scalars(f'results validation/train auc {mode} - last validation', {'lr: {}'.format(lrs[k]): sc}, trep)     
        
        return best_model, best_lr_scores[lr_idx], fb_time, best_lr_scores_per_class[lr_idx]
            

    def test_linear_classifier(self, model, dataloader, mode, save_roc_curves=False):
        model.eval()

        with torch.no_grad():
            labels = []
            out_probs = []
            for features, class_binary, class_multi, class_names, img_name  in dataloader:
                
                features = features.cuda() if torch.cuda.is_available() else features
                label = class_binary if mode == 'binary' else class_multi
                label = label.cuda() if torch.cuda.is_available() else label
                labels.append(label)

                # forward pass 
                out_raw = model(features)
                out_prob = torch.sigmoid(out_raw)
                out_probs.append(out_prob)
                
        y_pred = torch.cat(out_probs)
        y_pred = y_pred.cpu()
        y = torch.cat(labels)
        y = y.cpu()
        
        if self.debug_mode_on:
            if mode == 'binary':
                y[0] = 0.0
            else:
                y[0] = torch.tensor([float(i % 2 == 0) for i in range(y[0].shape[0])])
                y[1] = torch.tensor([float(i % 2 == 1) for i in range(y[0].shape[0])])
        
        if save_roc_curves:
        # save ROC curves
            class_names = [sublist[0] for sublist in class_names] 
            auc, path_roc = save_roc(y=y, y_pred_prob=y_pred , organ='liver', savepath=self.config['results_dir'], class_type=mode, name_classes=class_names)
            self.writer.add_image(f'results test/roc - {mode}', img_tensor = np.array(Image.open(path_roc)), dataformats='HWC')
            
        # save aucs 
        auc = roc_auc_score(y_true=y.numpy(), y_score=y_pred.numpy(), average='macro')
        if mode != "binary":
            auc_per_class = roc_auc_score(y_true=y.numpy(), y_score=y_pred.numpy(), average=None)
            auc_per_class = auc_per_class.tolist()
        else:
            auc_per_class = auc

        return auc, auc_per_class
    
    
    ######################################################################################################### 
    #                                                                                                       #
    #                                      HELPER FUNCTIONS                                                 #
    #                                                                                                       #
    ######################################################################################################### 
    
    
    def extract_and_save_embeddings(self, dataloader, model, savedir=None, eval_embedding_rank=False):
        model.eval()
        
        all_wsi_embs = []
        
        with torch.no_grad():
            for idx, (patch_emb, _, patch_positions, image_name, _, _, _) in enumerate(tqdm(dataloader, desc="Extract ssl features ...")):

                if self.debug_mode_on:
                    patch_emb = patch_emb[:,0:1000, :]
                    patch_positions = patch_positions[:, 0:1000, :]
                
                if isinstance(self.config['max_inf_tokens'], int):
                    if self.config['max_inf_tokens'] < patch_emb.shape[1]:
                        indices = torch.randint(0, patch_emb.shape[1], (self.config['max_inf_tokens'],))
                        patch_emb = patch_emb[:,indices,:]
                        patch_positions = patch_positions[:,indices,:]

                self.writer.add_scalar('params validation/tokens', patch_emb.shape[1], global_step=idx)

                patch_emb = patch_emb.cuda()
                
                patch_positions = patch_positions.cuda() 

                if isinstance(model, nn.DataParallel): 
                    wsi_emb = model.module.get_features(patch_emb, token_position_wsi=patch_positions)
                else:   
                    wsi_emb = model.get_features(patch_emb, token_position_wsi=patch_positions)
                    
                if not savedir:
                    savedir = os.path.join(self.config["results_dir"], f"temp")
                if not os.path.isdir(savedir):
                    os.mkdir(savedir)
                
                if eval_embedding_rank:
                    all_wsi_embs.append(wsi_emb.detach().cpu())    
                
                with h5py.File(os.path.join(savedir, image_name[0]), 'w') as hdf5_file:
                    hdf5_file.create_dataset("features", data=np.array(wsi_emb.detach().cpu()))    
        
        if eval_embedding_rank:
            all_wsi_embs = torch.cat(all_wsi_embs)
            rank = smooth_rank_measure(all_wsi_embs)
        else:
            rank = 0
            
        return savedir, rank
    
    def init_downstream_model(self, out_dim, in_dim): 
        model = nn.Linear(in_features=in_dim, out_features=out_dim)
        model.cuda() if torch.cuda.is_available() else model
        return model
    
    def init_loaders(self, config, train_split, val_split, test_split, curated_test_split):
        print('\nInit Loaders...\n', end=' ')
        train_loader = self.get_split_loader(train_split, training=True, batch_size=config['batch_size'])
        val_loader = self.get_split_loader(val_split,  training=False, batch_size=1)
        test_loader = self.get_split_loader(test_split, training=False, batch_size=1)
        curated_test_loader = self.get_split_loader(curated_test_split, training=False, batch_size=1)
        return train_loader,val_loader,test_loader, curated_test_loader
    
    def get_split_loader(self, dataset, training=False, batch_size=1):  
        kwargs = {'num_workers': self.config['num_workers']}
        if training:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                **kwargs
            )
        else:	
            dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                **kwargs)  
        return dataloader

    def get_optimizer(self, model, opt, lr):
        if opt == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) 
        elif opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        elif opt == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        else:
            raise NotImplementedError
        return optimizer

    def init_intra_wsi_loss_function(self):
        if self.config["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or self.config["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
            loss_fn = nn.MSELoss(reduction=self.config["mse_reduction"])
        else:
            loss_fn = InfoNCE(temperature=self.config["temperature"])
        return loss_fn
    
    def init_intra_rna_loss_function(self):
        loss_fn = nn.MSELoss(reduction=self.config["mse_reduction"])
        return loss_fn
    
    def init_inter_modality_loss_function(self):
        loss_fn = InfoNCE(temperature=self.config["temperature"])
        return loss_fn
    
    def apply_random_mask(self, patch_embeddings, patch_positions):
        _, dim_size, _ = patch_embeddings.shape
        mask_count = int(self.config["mask_percentage"] * dim_size)
        mask = torch.cat([torch.zeros(mask_count), torch.ones(dim_size - mask_count)])
        mask = mask[torch.randperm(dim_size)].unsqueeze(0).unsqueeze(-1)
        return patch_embeddings*mask, patch_positions*mask
        
    




