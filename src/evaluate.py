import os, glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import load_project_config
PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']
DATA_DIR = PROJECT_CONFIG['data_dir']

def agg_metrics():
    metric_files_pattern = os.path.join(PROJECT_DIR, 'models', '**', '**.csv')
    glob.glob(metric_files_pattern)

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

def get_performance_tables():
    metric_files_pattern = os.path.join(PROJECT_DIR, 'models', '**', '**.csv')
    metric_paths = glob.glob(metric_files_pattern)
    folds = tuple(
        pd.Series(metric_paths)
        .apply(lambda path: Path(path).stem)
        .str.extract('((?<=fold)\d)', expand=False)
        .fillna('all'))
    model_types = tuple(
        pd.Series(metric_paths)
        .apply(lambda path: Path(path).parent.stem))
    metrics_agg = pd.concat([get_metrics(path) for path in metric_paths], axis=1).T
    metrics_agg['Fold'] = folds
    metrics_agg['Model'] = model_types
    
    merged_model_metrics = (
        metrics_agg[metrics_agg.Fold=='all']
        .set_index('Model')
        .drop(columns=['Fold', 'name'])
        .sort_index()
    )
    metrics_agg = metrics_agg.drop(metrics_agg.index[metrics_agg.Fold=='all'])

#     model_overview = metrics_agg.drop(columns='name').set_index(['model', 'fold']).astype(float).sort_index()
    model_overview = (
        metrics_agg
        .drop(columns='name')
        .pivot(index='Fold', columns=['Model'])
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1))
    model_agg = metrics_agg.drop(columns='Fold').astype({
        'Accuracy': float,
        'F1 Score': float,
        'AUC': float
    }, errors='ignore').groupby(['Model']).mean().sort_index()
    
    return model_overview, model_agg, merged_model_metrics, metrics_agg