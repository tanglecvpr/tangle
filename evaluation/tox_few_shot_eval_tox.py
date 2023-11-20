#----> internal imports
from core.datasets.dataset_mmssl import SlideEmbeddingDataset
from core.utils.utils_mmssl import set_determenistic_mode

#----> general imports
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, auc
from tqdm import tqdm
from datetime import datetime
import numpy as np 
import os
        
##################### INPUT PARAMS #####################


feature_dirs_val = [
                   ]
feature_dirs_test = [dirs.replace('/val_split', '/test_split') for dirs in feature_dirs_val]


save_dir = ''

lesions = [
                'Cellular infiltration',
                'Fatty change',
                'Increased mitosis',
                'Hypertrophy',
                'Necrosis',
                'Proliferation',
        ]

exclude_gradings = None
csv_path = ""
few_shot_k = [1,5,10,25,50] # select k abnormal samples per class and k normal samples per class for training, if -1 all samples are used
seeds = [0,6,12,18,24] 
metrics = ["auc", "f1", "bacc"]
optimize_thresholds=False
max_iter=10000
treshold_value=0.9
########################################################



current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M-%S")

# Sanity check
for idx, path in enumerate(feature_dirs_val):
    if not os.path.isdir(path):
        raise ValueError(f"Invalid folder path in feature_dirs_val at position {idx}")
for idx, path in enumerate(feature_dirs_test):
    if not os.path.isdir(path):
        raise ValueError(f"Invalid folder path in feature_dirs_test at position {idx}")


results_df = pd.DataFrame()

print_flag_dataset = True
run_df_idx = 0
# iterate over different embedding spaces / models
for feature_dir_val, feature_dir_test in zip(feature_dirs_val, feature_dirs_test):
    
    model_name = feature_dir_val.split("/")[-2]
    
    feature_dataset_train = SlideEmbeddingDataset(feature_folder_path=feature_dir_val, 
                                                        dataset_csv=pd.read_csv(csv_path),
                                                        lesions=lesions, 
                                                        equal_normal_abnormal=False,
                                                        exclude_gradings=exclude_gradings,
                                                        do_testset_label_modifications=True
                                                        )

    feature_dataset_test = SlideEmbeddingDataset(feature_folder_path=feature_dir_test, 
                                                        dataset_csv=pd.read_csv(csv_path),
                                                        lesions=lesions, 
                                                        equal_normal_abnormal=False,
                                                        exclude_gradings=None,
                                                        do_testset_label_modifications=True
                                                        )
    
    # iterate over different k in few shot setting
    for k_idx, k in enumerate(tqdm(few_shot_k, desc=f"Experiment progress")):
        
        find_optimal_threshold_flag = True
        print_flag_thresholds = True
        lesion_thresholds = []
        
        # different runs with different data splits
        for run_idx, seed in enumerate(seeds):

            set_determenistic_mode(SEED=seed)
            
            if k==-1 or k==None or k=="all":
                few_shot_datasets_train = feature_dataset_train.get_few_shot_binary_datasets(lesions=lesions, k=None)
            else:
                few_shot_datasets_train = feature_dataset_train.get_few_shot_binary_datasets(lesions=lesions, k=k)
            
            few_shot_datasets_test = feature_dataset_test.get_few_shot_binary_datasets(lesions=lesions, k=None, test_flag=True)
            
            if print_flag_dataset:
                print(feature_dataset_train.print_dataset_summary())
                print(feature_dataset_test.print_dataset_summary())
                print_flag_dataset = False
                
            scores = []
            # train linear binary classifier seperately for each lesion
            for lesion_idx, lesion in enumerate(lesions):
                
                x_train = few_shot_datasets_train[lesion]["features"]
                y_train = few_shot_datasets_train[lesion]["binary_classes"].squeeze()
                x_test = few_shot_datasets_test[lesion]["features"]
                y_test = few_shot_datasets_test[lesion]["binary_classes"].squeeze()
                
                if run_idx == 0 and lesion_idx == 0:
                    print(f"\n################\nNumber of samples used for per-lesion {k}-shot training lesion {lesion}: {x_train.shape[0]}\nNumber of test samples used for per-lesion {k}-shot: {x_test.shape[0]}\ntesting lesion {lesion} with {np.count_nonzero(y_test)} samples\n################")
                
                clf = LogisticRegression(max_iter=max_iter).fit(x_train, y_train)
                
                if "bacc" in metrics:
                    y_pred_test_prob = clf.predict_proba(x_test)[:,1]
                    if optimize_thresholds:
                        fpr, tpr, thresholds = roc_curve(y_score=y_pred_test_prob, y_true=y_test, drop_intermediate=False)
                        # find optimal threshold for every tested model and few-shot k 
                        if find_optimal_threshold_flag:
                            baccs = []
                            for thresh in thresholds:
                                y_pred_test = np.where(y_pred_test_prob > thresh, 1, 0)
                                baccs.append(balanced_accuracy_score(y_pred=y_pred_test, y_true=y_test))
                            lesion_threshold = thresholds[np.argmax(baccs)]    
                            lesion_thresholds.append(lesion_threshold)
                            y_pred_test = np.where(y_pred_test_prob > lesion_threshold, 1, 0)
                        else:
                            y_pred_test = np.where(y_pred_test_prob > lesion_thresholds[lesion_idx], 1, 0) 
                    else:
                        y_pred_test = clf.predict(x_test)
                        y_pred_test = np.where(y_pred_test_prob > treshold_value, 1, 0)
                    y_pred_test = clf.predict(x_test)
                    results_df.loc[run_df_idx, "model_name"] = model_name
                    results_df.loc[run_df_idx, "run"] = run_idx
                    results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                    results_df.loc[run_df_idx, "k"] = f"{k}"
                    results_df.loc[run_df_idx, "metric"] = "BACC"
                    results_df.loc[run_df_idx, "score"] = balanced_accuracy_score(y_pred=y_pred_test, y_true=y_test)
                    run_df_idx += 1
                
                if "auc" in metrics:
                    y_pred_test_prob = clf.predict_proba(x_test)[:,1]
                    fpr, tpr, tresholds = roc_curve(y_score=y_pred_test_prob, y_true=y_test)
                    results_df.loc[run_df_idx, "model_name"] = model_name
                    results_df.loc[run_df_idx, "run"] = run_idx
                    results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                    results_df.loc[run_df_idx, "k"] = f"{k}"
                    results_df.loc[run_df_idx, "metric"] = "AUC"
                    results_df.loc[run_df_idx, "score"] = auc(fpr, tpr)
                    run_df_idx += 1
        
                if "f1" in metrics:
                    y_pred_test_prob = clf.predict_proba(x_test)[:,1]
                    if optimize_thresholds:
                        fpr, tpr, thresholds = roc_curve(y_score=y_pred_test_prob, y_true=y_test)
                        # find optimal threshold for every tested model and few-shot k 
                        if find_optimal_threshold_flag:
                            f1s = []
                            for thresh in thresholds:
                                y_pred_test = np.where(y_pred_test_prob > thresh, 1, 0)
                                f1s.append(f1_score(y_pred=y_pred_test, y_true=y_test))
                            lesion_threshold = thresholds[np.argmax(f1s)]    
                            lesion_thresholds.append(lesion_threshold)
                            y_pred_test = np.where(y_pred_test_prob > lesion_threshold, 1, 0)
                        else:
                            y_pred_test = np.where(y_pred_test_prob > lesion_thresholds[lesion_idx], 1, 0) 
                    else:
                        y_pred_test = np.where(y_pred_test_prob > treshold_value, 1, 0)
                    results_df.loc[run_df_idx, "model_name"] = model_name
                    results_df.loc[run_df_idx, "run"] = run_idx
                    results_df.loc[run_df_idx, "lesion"] = f"{lesion}"
                    results_df.loc[run_df_idx, "k"] = f"{k}"
                    results_df.loc[run_df_idx, "metric"] = "F1"
                    results_df.loc[run_df_idx, "score"] = f1_score(y_pred=y_pred_test, y_true=y_test)
                    run_df_idx += 1
            
            if optimize_thresholds:
                print(f"seed {seed} thresholds: {lesion_thresholds}")
            
            find_optimal_threshold_flag = False
            
    thres_str = '_thresoptim' if optimize_thresholds else ''
    results_df.to_csv(save_dir + "/" + formatted_time + "_" + f'{feature_dirs_test[0].split("/")[-1]}_' + str(metrics) + thres_str + ".csv")

thres_str = '_thresoptim' if optimize_thresholds else ''
results_df.to_csv(save_dir + "/" + formatted_time + "_" + f'{feature_dirs_test[0].split("/")[-1]}_' + str(metrics) + thres_str + ".csv")
            
        