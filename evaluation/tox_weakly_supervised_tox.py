#----> internal imports
from core.datasets.dataset_mmssl import DatasetFactory
from core.utils.utils_mmssl import set_determenistic_mode
from core.models.helpers import MLP
from core.models.abmil import ABMILEmbedder
from core.models.transmil import TransMIL
from core.utils.utils_mmssl import print_network

#----> pytorch imports
import torch
from torch.utils.data import DataLoader, Subset 

#----> general imports
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from tqdm import tqdm
from datetime import datetime
import numpy as np 
import time


##################### INPUT PARAMS #####################
patch_feature_dir = ''
csv_path = ""
split_file_path = ""
model_name = 'resnet50_weakly_supervised'
save_dir = ''
lesions = ['Cellular infiltration',
           'Fatty change',
           'Increased mitosis',
           'Hypertrophy',
           'Necrosis',
           'Proliferation',
        ]


# model_type = "TransMIL"
# params = {'input_dim': 768,
#             'hidden_dim': 768,
#             'heads': 1,
#             'num_landmarks': 256,
#             'dropout': 0.25
#                             }


model_type = "ABMIL"
params = {'pre_params': {'input_dim': 1024,
                         'hidden_dim': 768,
                         'leight_pre_attn': False,
                            },
         'attention_params': {'model': 'ABMIL',
                'params': {
                'input_dim': 768,
                'hidden_dim': 512,
                'dropout': True, 
                'activation': 'softmax',
                'n_classes': 1 
                    }
                    }
}


# model_type = "TransMIL"
# params = {'input_dim': 768,
#             'hidden_dim': 512,
#             'heads': 1,
#             'num_landmarks': 128,
#             'dropout': 0.25
#                             }


# model_type = "ABMIL"
# params = {'pre_params': {'input_dim': 768,
#                          'hidden_dim': 512,
#                          'leight_pre_attn': True,
#                             },
#          'attention_params': {'model': 'ABMIL',
#                 'params': {
#                 'input_dim': 512,
#                 'hidden_dim': 256,
#                 'dropout': True, 
#                 'activation': 'softmax',
#                 'n_classes': 1 
#                     }
#                     }
# }



eval_mode = "multilabel" # multilabel, binary
num_workers_inf=4
device_name = "cuda:2"
few_shot_k = [50] # [1,5,10,25,50] select k abnormal samples per class and k normal samples per class for training, if -1 all samples are used
seeds = [0,6,12,18,24]  # [0,6,12,18,24] 
metrics = ["auc", "f1", "bacc"]
learning_rates = [0.0001] # [0.0001, 0.0005, 0.00001]
epochs = 30
print_val_performance = False
eval_testset = True
########################################################

model_name = model_name + f"_{model_type}"

device = torch.device(f"{device_name}" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, epochs, output_dim, val_dataloader=None, lesion=None, print_flag_data=False, print_flag_model=False):
    
    
    if model_type == "ABMIL":
        model = ABMILEmbedder(pre_attention_params=params["pre_params"],
                        attention_params=params["attention_params"]
                        )
        # ds_model = CLSS(input_dim=params["attention_params"]["params"]["input_dim"], output_dim=output_dim)
        ds_model = MLP(input_dim=params["attention_params"]["params"]["input_dim"], output_dim=output_dim)
    elif model_type == "TransMIL":
        model = TransMIL(input_dim=params["input_dim"],
                         hidden_dim=params["hidden_dim"],
                         heads=params["heads"],
                         num_landmarks=params["num_landmarks"],
                         dropout=params["dropout"],
                         device=device)
        # ds_model = CLSS(input_dim=params["hidden_dim"], output_dim=output_dim)
        ds_model = MLP(input_dim=params["input_dim"], output_dim=output_dim)
    else:
        raise ValueError("Invalid model_type specification")
    
    
    if print_flag_model:
        print_network(model)
        print_network(ds_model)
    
    optimizer = torch.optim.AdamW(list(model.parameters())+list(ds_model.parameters()), lr=lr)
    
    model.to(device)
    ds_model.to(device)
    model.train()
    ds_model.train()
    
    log_interval = 5
    
    # train weakly supervised classifier for binary case
    for _, epoch in enumerate(range(epochs)):
        
        losses = []
        for batch_idx, (patch_emb, _, _, slide_info, _, _, _) in enumerate(train_dataloader):

            if print_flag_data and epoch==0 and batch_idx==0:
                print(f"\n################\nNumber of samples used for {k}-shot training lesion {lesion}: {len(train_dataloader)}\n################\n")
            
            y = train_dataloader.dataset.dataset.dataset_csv[train_dataloader.dataset.dataset.dataset_csv["IMAGE_NAME"]==slide_info[0]][lesions]
            
            if lesion:
                y = torch.tensor(y[lesion].to_numpy(), dtype=torch.float32).unsqueeze(dim=0).to(device) 
            else:
                y = torch.tensor(y.to_numpy(), dtype=torch.float32).to(device) 
            
            patch_emb = patch_emb.to(device)
            
            y_emb = model(patch_emb)
            
            y_pred = ds_model(y_emb)
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
            
            losses.append(loss.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        ep_loss = sum(losses) / len(train_dataloader)
        if epoch%log_interval == 0:
            print(f"Epoch {epoch} {lesion} Loss: {ep_loss}")
            if val_dataloader is not None:
                y_pred_prob_test, y_tests = eval(model=model, ds_model=ds_model, dataloader=val_dataloader, lesion=lesion)
                model.train()
                ds_model.train()
                print(f"Epoch {epoch} {lesion} Eval AUC: {roc_auc_score(y_score=y_pred_prob_test, y_true=y_tests)}")    
        
    return model, ds_model

def eval(model, ds_model, dataloader, lesion=None, print_flag_data=False):
    model.to(device)
    ds_model.to(device)
    with torch.no_grad():
        # evaluate the binary weakly supervised model
        model.eval()
        ds_model.eval()
        y_pred_prob_test = []
        y_tests = []
        
        for batch_idx, (patch_emb, _, _, slide_info, _, _, _) in enumerate(dataloader):
            y_test = dataloader.dataset.dataset_csv[dataloader.dataset.dataset_csv["IMAGE_NAME"]==slide_info[0]][lesions]
            
            if lesion:
                y_test = torch.tensor(y_test[lesion].to_numpy(), dtype=torch.float32).squeeze().to(device) 
            else:
                y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).squeeze().to(device)
            
            if print_flag_data and batch_idx==0:
                print(f"\n################\nNumber of samples used for {k}-shot testing lesion {lesion}: {len(dataloader)}\n################\n")    
            
            patch_emb = patch_emb.to(device)
            
            y_emb = model(patch_emb)
            y_prob = torch.sigmoid(ds_model(y_emb)).squeeze()
            
            y_pred_prob_test.append(y_prob.cpu())
            y_tests.append(y_test.cpu())
            
        y_pred_prob_test = np.stack(y_pred_prob_test)
        y_tests = np.stack(y_tests)
        
    return y_pred_prob_test, y_tests

current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M-%S")

results = pd.DataFrame()
results_indices = []

results_df = pd.DataFrame()
    
dataset = DatasetFactory(patch_feature_dir=patch_feature_dir,
                            csv_path=csv_path,
                            rnaseq_path=None,
                            split_file_path=split_file_path,
                            n_tokens=-1,
                            
                            )

patch_feature_dataset_train = dataset.return_val_split()
patch_feature_dataset_val = dataset.return_cur_test_split()
patch_feature_dataset_test = dataset.return_test_split()


run_df_idx = 0
print_flag_model = False
print_flag_data = False
print_flag_data_test = False
# iterate over different k in few shot setting
for k_idx, k in enumerate(tqdm(few_shot_k, desc=f"Whole progress")):
    
    lesion_thresholds = []
    
    # different runs with different data splits
    for run_idx, seed in enumerate(tqdm(seeds, desc='Run progress')):

        set_determenistic_mode(SEED=seed)
        
        if eval_mode == "multilabel":
            if k==-1 or k==None or k=="all":
                subset_indices = patch_feature_dataset_train.subsample_k_shot_indices(k=None, lesions=lesions)
                few_shot_datasets_train = Subset(patch_feature_dataset_train, subset_indices)
            else:
                subset_indices = patch_feature_dataset_train.subsample_k_shot_indices(k=k, lesions=lesions)
                few_shot_datasets_train = Subset(patch_feature_dataset_train, subset_indices)
            
            train_dataloader = DataLoader(few_shot_datasets_train, batch_size=1, shuffle=True)
            val_dataloader = DataLoader(patch_feature_dataset_val, batch_size=1, num_workers=num_workers_inf)
            test_dataloader = DataLoader(patch_feature_dataset_test, batch_size=1, num_workers=num_workers_inf)
            
        test_aucs = []
        test_baccs = []
        test_f1_scores = []
        models = []
        ds_models = []    
        
        # train a binary MIL model for each lesion seperately
        if eval_mode == 'binary':
            
            for lesion_idx, lesion in enumerate(lesions):
                          
                if k==-1 or k==None or k=="all":
                    subset_indices = patch_feature_dataset_train.subsample_k_shot_indices(k=None, lesions=[lesion])
                    few_shot_datasets_train = Subset(patch_feature_dataset_train, subset_indices)
                else:
                    subset_indices = patch_feature_dataset_train.subsample_k_shot_indices(k=k, lesions=[lesion])
                    few_shot_datasets_train = Subset(patch_feature_dataset_train, subset_indices)
            
                train_dataloader = DataLoader(few_shot_datasets_train, batch_size=1, shuffle=True)
                val_dataloader = DataLoader(patch_feature_dataset_val, batch_size=1, num_workers=num_workers_inf)
                test_dataloader = DataLoader(patch_feature_dataset_test, batch_size=1, num_workers=num_workers_inf)
                
                for lr_idx, lr in enumerate(learning_rates):
                    
                    if k_idx==0 and run_idx==0 and lr_idx==0 and lesion_idx==0:
                        print_flag_model = True
                    else:
                        print_flag_model = False 
                    if run_idx==0 and lr_idx==0 and lesion_idx==0:
                        print_flag_data = True
                    else:
                        print_flag_data = False
                    
                    model, ds_model = train(epochs=epochs, 
                                            train_dataloader=train_dataloader, 
                                            lesion=lesion,
                                            output_dim=1, 
                                            print_flag_data=print_flag_data, 
                                            print_flag_model=print_flag_model,
                                            val_dataloader=val_dataloader if print_val_performance else None)
                    
                    # Choose best model on validation split 
                    s = time.time()
                    y_pred_prob_test, y_tests = eval(model=model, 
                                                     ds_model=ds_model, 
                                                     dataloader=val_dataloader,
                                                     lesion=lesion
                                                     )
                    e = time.time()
                    print(f"Inference time val: {(e-s)/60} min")
                    test_aucs.append(roc_auc_score(y_score=y_pred_prob_test, y_true=y_tests))
                    print(f"Val AUC {lesion} k={k}: {test_aucs[-1]}")
                    models.append(model.cpu())
                    ds_models.append(ds_model.cpu())
            
                best_lr_idx = np.argmax(test_aucs)
                test_model = models[best_lr_idx]
                test_model_ds = ds_models[best_lr_idx]

                if run_idx==0 and lesion_idx==0:
                    print_flag_data_test = True
                else:
                    print_flag_data_test = False
                
                if eval_testset:                        
                    s = time.time()
                    y_pred_prob_test, y_tests = eval(model=test_model, 
                                                        ds_model=test_model_ds, 
                                                        dataloader=test_dataloader,
                                                        lesion=lesion,
                                                        print_flag_data=print_flag_data_test)
                    e = time.time()
                    print(f"Inference time test: {(e-s)/60} min")
                    print(f"Test AUC {lesion} k={k}: {roc_auc_score(y_score=y_pred_prob_test, y_true=y_tests)}")
                    
                    y_pred_class_test = np.where(y_pred_prob_test > 0.5, 1, 0)
                    
                    if "bacc" in metrics:
                        results_df.loc[run_df_idx, "model_name"] = model_name
                        results_df.loc[run_df_idx, "run"] = run_idx
                        results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                        results_df.loc[run_df_idx, "k"] = f"{k}"
                        results_df.loc[run_df_idx, "metric"] = "BACC"
                        results_df.loc[run_df_idx, "score"] = balanced_accuracy_score(y_pred=y_pred_class_test, y_true=y_tests)
                        run_df_idx += 1
                    
                    if "auc" in metrics:
                        fpr, tpr, tresholds = roc_curve(y_score=y_pred_prob_test, y_true=y_tests)
                        results_df.loc[run_df_idx, "model_name"] = model_name
                        results_df.loc[run_df_idx, "run"] = run_idx
                        results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                        results_df.loc[run_df_idx, "k"] = f"{k}"
                        results_df.loc[run_df_idx, "metric"] = "AUC"
                        results_df.loc[run_df_idx, "score"] = auc(fpr, tpr)
                        run_df_idx += 1

                    if "f1" in metrics:
                        results_df.loc[run_df_idx, "model_name"] = model_name
                        results_df.loc[run_df_idx, "run"] = run_idx
                        results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                        results_df.loc[run_df_idx, "k"] = f"{k}"
                        results_df.loc[run_df_idx, "metric"] = "F1"
                        results_df.loc[run_df_idx, "score"] = f1_score(y_pred=y_pred_class_test, y_true=y_tests)
                        run_df_idx += 1
                                
                results_df.to_csv(save_dir + "/" + formatted_time + 'wsl_' + f'{eval_mode}_' + f'lr{learning_rates}_' + f'ep{epochs}_' + str(metrics) + ".csv")

                
        # train a multilabel MIL model with k*len(lesions) samples 
        elif eval_mode == 'multilabel': 
            # grid search over learning rates
            for lr_idx, lr in enumerate(learning_rates):
            
                if k_idx==0 and run_idx==0 and lr_idx==0:
                    print_flag_model = True 
                if run_idx==0 and lr_idx==0:
                    print_flag_data = True
                
                model, ds_model = train(epochs=epochs, 
                                        train_dataloader=train_dataloader, 
                                        output_dim=len(lesions), 
                                        print_flag_data=print_flag_data, 
                                        print_flag_model=print_flag_model,
                                        val_dataloader=val_dataloader if print_val_performance else None)
                        
                # Choose best model on validation split in multi-label setup
                s = time.time()
                y_pred_prob_test, y_tests = eval(model=model, 
                                                 ds_model=ds_model, 
                                                 dataloader=val_dataloader)
                e = time.time()
                print(f"Inference time val: {(e-s)/60} min")
                test_aucs.append(roc_auc_score(y_score=y_pred_prob_test, y_true=y_tests, average='macro'))
                print(f"Val Macro AUC k={k}: {test_aucs[-1]}")
                models.append(model.cpu())
                ds_models.append(ds_model.cpu())

            best_lr_idx = np.argmax(test_aucs)
            test_model = models[best_lr_idx]
            test_model_ds = ds_models[best_lr_idx]
            
            if eval_testset:
                s = time.time()
                y_pred_prob_test, y_tests = eval(model=test_model, 
                                                    ds_model=test_model_ds, 
                                                    dataloader=test_dataloader)
                e = time.time()
                print(f"Inference time test: {(e-s)/60} min")
                print(f"Test Macro AUC k={k}: {roc_auc_score(y_score=y_pred_prob_test, y_true=y_tests)}")
            
            
                for lesion_idx, lesion in enumerate(lesions):

                    # transform to binary probs/classes
                    y_bin_class = y_tests[:,lesion_idx]
                    y_pred_bin_prob = y_pred_prob_test[:, lesion_idx]
                    y_pred_bin_class = np.where(y_pred_bin_prob > 0.5, 1, 0)
                    
                    if "bacc" in metrics:
                        results_df.loc[run_df_idx, "model_name"] = model_name
                        results_df.loc[run_df_idx, "run"] = run_idx
                        results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                        results_df.loc[run_df_idx, "k"] = f"{k}"
                        results_df.loc[run_df_idx, "metric"] = "BACC"
                        results_df.loc[run_df_idx, "score"] = balanced_accuracy_score(y_pred=y_pred_bin_class, y_true=y_bin_class)
                        run_df_idx += 1
                    
                    if "auc" in metrics:
                        fpr, tpr, tresholds = roc_curve(y_score=y_pred_bin_prob, y_true=y_bin_class)
                        results_df.loc[run_df_idx, "model_name"] = model_name
                        results_df.loc[run_df_idx, "run"] = run_idx
                        results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                        results_df.loc[run_df_idx, "k"] = f"{k}"
                        results_df.loc[run_df_idx, "metric"] = "AUC"
                        results_df.loc[run_df_idx, "score"] = auc(fpr, tpr)
                        run_df_idx += 1

                    if "f1" in metrics:
                        results_df.loc[run_df_idx, "model_name"] = model_name
                        results_df.loc[run_df_idx, "run"] = run_idx
                        results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                        results_df.loc[run_df_idx, "k"] = f"{k}"
                        results_df.loc[run_df_idx, "metric"] = "F1"
                        results_df.loc[run_df_idx, "score"] = f1_score(y_pred=y_pred_bin_class, y_true=y_bin_class)
                        run_df_idx += 1
                                    
                    results_df.to_csv(save_dir + "/" + formatted_time + 'wsl_' + f'{eval_mode}_' + f'lr{learning_rates}_' + f'ep{epochs}_' + str(metrics) + ".csv")
        
    
        else:
            raise ValueError('Invalid eval_mode')
        

results_df.to_csv(save_dir + "/" + formatted_time + 'wsl_' + f'{eval_mode}_' + f'lr{learning_rates}_' + f'ep{epochs}_' + str(metrics) + ".csv")
                   
                
                
                
            
        