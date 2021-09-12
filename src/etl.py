from eda import load_label_map
import random
import pandas as pd
import os
from utils import load_project_config

PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']
DATA_DIR = PROJECT_CONFIG['data_dir']
SPLIT_MAP_OUTPATH = os.path.join(PROJECT_DIR, 'data', 'cleaned', 'split_map.csv')
SAMPLING_RATE = 250

def threeway_split(tup):
    start, stop = tup
    spread = (stop - start) // 3
    first_stop = start + spread
    second_start = first_stop + 1
    second_stop = second_start + spread
    third_start = second_stop
    return [(start, first_stop), (second_start, second_stop), (third_start, stop)]

def train_test_split(lst):
    pop_idx = random.randint(0, 2)
    train = lst.pop(pop_idx)
    return train, lst[0], lst[1]

def generate_train_test_split_map():
    # select usable samples, recompute duration/end
    df = load_label_map()
    duration_secs = df.duration / SAMPLING_RATE
    remainder_3 = df.duration % 3
    df.end = df.end - remainder_3
    df.duration = df.duration - remainder_3
    usable_samples = (df.duration / SAMPLING_RATE) >= 30.0
    df = df[usable_samples]
    
    # 3-fold split
    idx_gb = df.groupby(['annot', 'record']).apply(lambda row: [tup for tup in zip(row.start, row.end)])
    idx_gb = idx_gb.apply(lambda lst: [train_test_split(threeway_split(tup)) for tup in lst])
    idx_gb = idx_gb.explode()
    
    # decouple train and test sets
    test = idx_gb.apply(lambda tup: tup[0])
    test.name = 'range'
    test = test.to_frame().reset_index()

    train_1 = idx_gb.apply(lambda tup: tup[1])
    train_1.name = 'range'
    train_1 = train_1.to_frame().reset_index()

    train_2 = idx_gb.apply(lambda tup: tup[2])
    train_2.name = 'range'
    train_2 = train_2.to_frame().reset_index()
    
    # add labels
    test['split'] = 'test'
    train_1['split'] = 'train'
    train_2['split'] = 'train'
    
    # merge into df with split
    split_df = pd.concat([test, train_1, train_2], ignore_index=True)
    start = split_df.range.apply(lambda tup: tup[0])
    end = split_df.range.apply(lambda tup: tup[1])
    split_df.drop(columns='range', inplace=True)
    split_df['start'] = start 
    split_df['end'] = end
    
    # drop AFL and J (not enough data)
    split_df = split_df[split_df.annot.isin({'AFIB', 'N'})]
    split_df.to_csv(SPLIT_MAP_OUTPATH, index=False)
    
def load_split_map():
    return pd.read_csv(SPLIT_MAP_OUTPATH, dtype={"record": str})