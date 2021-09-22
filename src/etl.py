from eda import load_label_map
import random
import pandas as pd
import numpy as np
import os
from utils import load_project_config

PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']
DATA_DIR = PROJECT_CONFIG['data_dir']
SPLIT_MAP_OUTPATH = os.path.join(PROJECT_DIR, 'data', 'cleaned', 'split_map.csv')
SPLIT_MAP_EXPANDED_OUTPATH = os.path.join(PROJECT_DIR, 'data', 'cleaned', 'split_map_expanded.csv')
SAMPLING_RATE = 250

def load_split_map():
    return pd.read_csv(SPLIT_MAP_OUTPATH, dtype={"record": str})

def load_expanded_split_map():
    return pd.read_csv(SPLIT_MAP_EXPANDED_OUTPATH, dtype={"record": str})

def random_threeway_split(tup):
    start, stop = tup
    spread = (stop - start) // 3
    first_stop = start + spread
    second_start = first_stop + 1
    second_stop = second_start + spread
    third_start = second_stop
    
    groups = [(start, first_stop), (second_start, second_stop), (third_start, stop)]
    np.random.shuffle(groups)
    
    return tuple(groups)

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
    idx_gb = idx_gb.apply(lambda lst: [random_threeway_split(tup) for tup in lst])
    idx_gb = idx_gb.explode()
    
    # decouple train and test sets
    split_1 = idx_gb.apply(lambda tup: tup[0])
    split_1.name = 'range'
    split_1 = split_1.to_frame().reset_index()

    split_2 = idx_gb.apply(lambda tup: tup[1])
    split_2.name = 'range'
    split_2 = split_2.to_frame().reset_index()

    split_3 = idx_gb.apply(lambda tup: tup[2])
    split_3.name = 'range'
    split_3 = split_3.to_frame().reset_index()
    
    # add labels
    split_1['split'] = 1
    split_2['split'] = 2
    split_3['split'] = 3
    
    # merge into df with split
    split_df = pd.concat([split_1, split_2, split_3], ignore_index=True)
    start = split_df.range.apply(lambda tup: tup[0])
    end = split_df.range.apply(lambda tup: tup[1])
    split_df.drop(columns='range', inplace=True)
    split_df['start'] = start 
    split_df['end'] = end
    split_df['duration'] = split_df.end - split_df.start
    
    # OHE Split columns for 3-fold cross-val
    split_df['split_1'] = split_df.split == 1
    split_df['split_2'] = split_df.split == 2
    split_df['split_3'] = split_df.split == 3
    
    # drop AFL and J (not enough data)
    split_df = split_df[split_df.annot.isin({'AFIB', 'N'})]
    split_df.to_csv(SPLIT_MAP_OUTPATH, index=False)

def explode_split_map():
    def explode_sample(row):
        # compute valid starting indexes
        half_sig_len = 2500 // 2
        duration = row.end - row.start
        starts = np.arange(duration // half_sig_len - 1)
        starts *= half_sig_len
        starts += row.start
#         starts = np.arange(row.start, row.end - half_sig_len, half_sig_len)
        df = pd.DataFrame(
            np.vstack((
                np.repeat(row.annot, starts.size),
                np.repeat(row.record, starts.size),
                np.repeat(row.split, starts.size),
                starts,
                (starts + 2500),
                np.repeat(row.split_1, starts.size),
                np.repeat(row.split_2, starts.size),
                np.repeat(row.split_3, starts.size)
            )).T,
            columns = split_map.loc[2].index.drop('duration')
        )
        return df
        
    split_map = load_split_map()
    header = True
    
    for idx, row in split_map.iterrows():
        exploded_df = explode_sample(row)
        if header:
            exploded_df.to_csv(SPLIT_MAP_EXPANDED_OUTPATH, index=False)
            header = False
        else:
            exploded_df.to_csv(SPLIT_MAP_EXPANDED_OUTPATH, index=False, header=False, mode='a')
        
    
    