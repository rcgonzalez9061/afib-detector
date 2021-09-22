import os, glob, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ExpandedAfibDataset # takes a long time
from models import Hsieh2020, Afib_CNN, merge_models, Afib_CNN
from mobilenetv2 import MobileNetV2
from utils import load_model
import json

from utils import load_project_config
PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']
DATA_DIR = PROJECT_CONFIG['data_dir']


def eval_saved_models():
    eval_merged_models()
    
    model_pattern = os.path.join(PROJECT_DIR, 'models', '**', '**.pt')
    model_paths = glob.glob(model_pattern)
    folds = tuple(
        pd.Series(model_paths)
        .apply(lambda path: Path(path).stem)
        .str.extract('((?<=fold)\d)', expand=False)
        .fillna('all')
    )

    for model_path, fold in zip(model_paths, folds): 
        model_type = Path(model_path).parent.stem
        args_path = os.path.join(Path(model_path).parent, 'model_args.json')
        with open(args_path, 'r') as args_file:
            model_args = json.load(args_file)
        
        print(f'Evaluating {Path(model_path).stem}...')
        if model_type == 'Hsieh':
            eval_hsieh(model_path, fold, model_args)
        elif model_type == 'MobileNetV2':
            eval_mobilenet(model_path, fold, model_args)
        elif model_type == 'Custom':
            eval_custom_model(model_path, fold, model_args)
    
def eval_custom_model(
    model_path,
    fold,
    model_args,
    no_cuda=False,
    test_batch_size=1000
):
    eval_model(
        model_path, Afib_CNN,
        model_args,
        fold=fold,
        no_cuda=no_cuda,
        test_batch_size=test_batch_size
    )
    
def eval_merged_models(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    models_dir = os.path.join(PROJECT_DIR, 'models')
    model_folders = glob.glob(os.path.join(models_dir, '*'))
    
    for model_folder in model_folders:
        model_name = Path(model_folder).stem
        print(f'Evaluating {model_name} merged...')
        merged_outpath = os.path.join(model_folder, 'merged', f'{model_name} merged.pt')
        eval_outpath = os.path.join(model_folder, f'{model_name} merged.csv')
        if os.path.exists(merged_outpath): # delete existing merged files
            os.remove(merged_outpath)
        if model_name == 'Hsieh':
            model_class = Hsieh2020
        elif model_name == 'MobileNetV2':
            model_class = MobileNetV2
        elif model_name == 'Custom':
            model_class = Afib_CNN
        
        args_path = os.path.join(model_folder, 'model_args.json')
        with open(args_path, 'r') as args_file:
            model_args = json.load(args_file)
        model_pattern = os.path.join(model_folder, '*.pt')
        merged_model = merge_models(model_class, merged_outpath, device, *glob.glob(model_pattern), **model_args)
        eval_merged_model(merged_model, eval_outpath, no_cuda)
        
def eval_merged_model(
    model,
    eval_outpath,
    no_cuda=False,
    test_batch_size=1000,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    _, eval_dataset = ExpandedAfibDataset.load_train_test_datasets(
        random_seed=42,
        fold='all'
    )
    
    softmax, preds, groundtruth, idx = _eval_model(model, eval_dataset, device, test_batch_size)
    softmax = np.concatenate(softmax)
    preds = np.concatenate(preds).T[0]
    groundtruth = np.concatenate(groundtruth).T
    
    df = pd.DataFrame()
    df['argmax'] = preds
    df['exp_log_softmax_0'] = np.exp(softmax[:, 0]) # list(softmax)
    df['exp_log_softmax_1'] = np.exp(softmax[:, 1]) # list(softmax)
    df['truth'] = groundtruth
    df = df.set_index(np.array(idx))
    df.to_csv(eval_outpath)
            
def eval_hsieh(
    model_path,
    fold,
    model_args,
    no_cuda=False,
    test_batch_size=1000
):
    eval_model(
        model_path, Hsieh2020,
        model_args,
        fold=fold,
        no_cuda=no_cuda,
        test_batch_size=test_batch_size
    )

def eval_mobilenet(
    model_path,
    fold,
    model_args,
    no_cuda=False,
    test_batch_size=1000
):
    eval_model(
        model_path, MobileNetV2,
        model_args,
        fold=fold,
        no_cuda=no_cuda,
        test_batch_size=test_batch_size)
    
def eval_model(
    model_path,
    model_class,
    model_args,
    fold=0,
    no_cuda=False,
    test_batch_size=1000
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    _, eval_dataset = ExpandedAfibDataset.load_train_test_datasets(
        random_seed=42,
        fold=fold
    )
    
    model = load_model(model_path, model_class, model_args, device)
    softmax, preds, groundtruth, idx = _eval_model(model, eval_dataset, device, test_batch_size)
    softmax = np.concatenate(softmax)
    preds = np.concatenate(preds).T[0]
    groundtruth = np.concatenate(groundtruth).T
    
    pred_out = model_path[:-3] + '.csv'
    df = pd.DataFrame()
    df['argmax'] = preds
    df['exp_log_softmax_0'] = np.exp(softmax[:, 0]) # list(softmax)
    df['exp_log_softmax_1'] = np.exp(softmax[:, 1]) # list(softmax)
    df['truth'] = groundtruth
    df = df.set_index(np.array(idx))
    df.to_csv(pred_out)

def _eval_model(
    model,
    dataset,
    device, 
    test_batch_size=1000,
):
    model.eval()
    test_loss = 0
    correct = 0
    test_size = len(dataset)
    
    softmax = []
    preds = []
    groundtruth = []
    idxs = []
    
    with torch.no_grad():
        num_remaining = test_size
        while num_remaining > 0:
            batch_size = min((num_remaining, test_batch_size))
            idx, data, target = dataset.get_batch(batch_size)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_remaining -= batch_size
            
            softmax.append(output.cpu().numpy())
            preds.append(pred.cpu().numpy())
            groundtruth.append(target.cpu().numpy())
            idxs.extend(idx)

    
    avg_loss = test_loss / test_size
    accuracy = 100. * correct / test_size
    return (softmax, preds, groundtruth, idxs)

def get_metrics(model_eval_csv):
    model_name = Path(model_eval_csv).stem
    preds_df = pd.read_csv(model_eval_csv, index_col=0)
    
    accuracy = accuracy_score(preds_df.truth, preds_df.argmax)
    model_f1_score = f1_score(preds_df.truth, preds_df.argmax)
    roc_auc = roc_auc_score(preds_df.truth, preds_df.exp_log_softmax_1)
    
    metrics = pd.Series({
        'name': model_name,
        'Accuracy': accuracy,
        'F1 Score': model_f1_score,
        'AUC': roc_auc
    })
    return metrics

def main():
    eval_saved_models()

if __name__=='__main__':
    main()