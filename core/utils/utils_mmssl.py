#----> general imports
import numpy as np
import os
from datetime import datetime
import sklearn
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random
import copy

#----> torch imports 
import torch
import torch.backends.cudnn
import torch.cuda

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_args():
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--results_dir', type=str, default=None, help='Path to directory where results are saved')
    args = parser.parse_args() 
    return args

def smooth_rank_measure(embedding_matrix, eps=1e-7):
    """
    Compute the smooth rank measure of a matrix of embeddings.
    
    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension
        eps (float): Smoothing parameter to avoid division by zero.

    Returns:
        float: Smooth rank measure.
    """
    
    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)
    
    # Compute the smooth rank measure
    p = S / torch.norm(S, p=1) + eps
    p = p[:embedding_matrix.shape[1]]
    smooth_rank = torch.exp(-torch.sum(p * torch.log(p)))
    smooth_rank = round(smooth_rank.item(), 2)
    
    return smooth_rank


def get_custom_result_dir(config):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)
    """
    param_code = ''

    # ----> Time Stamp to make it unique
    param_code += '%s' % datetime.now().strftime("%d%m_%H%M%S")

    param_code += '%s' % ''.join(config["exp_name"])

    # ----> Updating
    config["results_dir"] = os.path.join(config["results_dir"], param_code)

    return config

def check_config_entries(config, config_idx):

    if config["patch_feature_dir"] is not None:
        if not os.path.isdir(config["patch_feature_dir"]):
            raise ValueError("Invalid path for patch_feature_dir at position {}.".format(config_idx))
        
    if config["rnaseq_path"] is not None:
        if not os.path.isfile(config["rnaseq_path"]):
            raise ValueError("Invalid path for rnaseq_path at position {}.".format(config_idx))

    if config["csv_path"] is not None:
        if not os.path.isfile(config["csv_path"]):
            raise ValueError("Invalid path for csv_path at position {}.".format(config_idx))
        

def get_single_config(config):
    configs = []
    for i in range(len(config["exp_name"])):
        cfg = copy.deepcopy(config)
        for key in config.keys():
            if type(config[key]) == list and key != 'gpu_devices' and key != 'ds_learning_rates' and key != "lesions":
                try:
                    cfg[key] = config[key][i]
                except IndexError:
                    raise IndexError("Check key: {}".format(key))
        configs.append(cfg)
    return configs

def set_determenistic_mode(SEED, disable_cudnn=False):
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)  # Set python seed for custom operators.
    rs = RandomState(
        MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(
        SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False  # Causes cuDNN to deterministically select an algorithm,
        
        torch.backends.cudnn.deterministic = True  # Causes cuDNN to use a deterministic convolution algorithm,
    else:
        torch.backends.cudnn.enabled = False  # Controls whether cuDNN is enabled or not.
        # If you want to enable cuDNN, set it to True.

  
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=200, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, score, model):

        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            return True 
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_model = model
            self.counter = 0
            return True
        

def print_network(net, results_dir=None):
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    if results_dir is not None:
        fname = "model_" + results_dir.split("/")[-1] + ".txt"
        path = os.path.join(results_dir, fname)
        f = open(path, "w")
        f.write(str(net))
        f.write("\n")
        f.write('Total number of parameters: %d \n' % num_params)
        f.write('Total number of trainable parameters: %d \n' % num_params_train)
        f.close()

    print(net)
    
    
def save_roc(y, y_pred_prob, organ, savepath, class_type, name_classes):

    auc = roc_auc_score(y_true=y.numpy(), y_score=y_pred_prob.numpy(), average='macro')
    
    if class_type == "binary":
        # plot and save ROC curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y, y_score=y_pred_prob)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        disp_roc = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='{}'.format(
            "Kidney Binary" if organ == "kidney" else "Liver Binary"))
        disp_roc.plot(ax=ax)
        ax.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
        disp_roc.figure_.suptitle("ROC")
        path_roc = os.path.join(savepath, "roc_{}_binary.png".format(organ))
        disp_roc.figure_.savefig(path_roc, bbox_inches='tight', format="png")
    else:
        # Compute ROC curve and ROC AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(name_classes)):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y[:, i], y_pred_prob[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        # Set up the figure and axis for the plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # Set up colors and markers for plotting
        colors = cycle(
            ['#1f77b4', '#7da7d9', '#729ece', '#2a9df4', '#1c9af4', '#1689cc', '#59a4b4', '#4b97c4', '#2a8ea4',
             '#367da0', '#2b7b99', '#23648e'])
        linestyles = cycle(['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'])
        markers = cycle(['o', 'v', '^', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd'])

        # Plot ROC curve for each class
        for i, color, marker, linestyle in zip(range(len(name_classes)), colors, markers, linestyles):
            ax.plot(fpr[i], tpr[i], color=color, marker=marker, lw=2, linestyle=linestyle,
                     label='ROC {0} (macro AUC = {1:0.2f})'.format(name_classes[i], roc_auc[i]))

        # Customize the plot
        ax.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
        ax.legend(loc='upper center', fancybox=True, ncol=3, bbox_to_anchor =(0.5,-0.2))
        ax.grid(True)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.suptitle("One vs. Rest ROC - macro AUC: {}".format(np.round(auc, 3)))

        # Save the figure
        path_roc = os.path.join(savepath, "roc_{}_multi.png".format(organ))
        fig.savefig(path_roc, bbox_inches='tight', format="png")

    plt.close("all")
    return auc, path_roc
