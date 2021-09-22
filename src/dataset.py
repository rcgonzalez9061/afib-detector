import numpy as np
import torch

from etl import load_split_map, load_expanded_split_map
from wfdb_ext import RecordCollection

from utils import load_project_config

PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']
DATA_DIR = PROJECT_CONFIG['data_dir']

LABEL_MAP = {
    'N': 0,
    'AFIB': 1,
    'AFL': 2,
    'J': 3
}
INV_LABEL_MAP = {item: key for key, item in LABEL_MAP.items()}
RECORD_COLLECTION = RecordCollection(DATA_DIR) # cache all records

class ExpandedAfibDataset():
    def __init__(self, sample_df, random_seed=None, transform=None):
        self.sample_df = sample_df
        self.transform = transform
        self.random_state = np.random.RandomState(random_seed)
        
        self.shuffled_idx = list(self.sample_df.index)
        self.random_state.shuffle(self.shuffled_idx)
    
    def __len__(self):
        return self.sample_df.shape[0]
    
    def reset(self):
        self.shuffled_idx = list(self.sample_df.index)
        self.random_state.shuffle(self.shuffled_idx)
        
    def get_batch(self, size, as_tensor=True):
        if not self.shuffled_idx:
            self.reset()
        
        if len(self.shuffled_idx) <= size:
            batch_idx = self.shuffled_idx
            self.shuffled_idx = []
        else:
            batch_idx = [self.shuffled_idx.pop() for i in range(size)]
        batch = self.sample_df.loc[batch_idx]
        samples = batch.apply(
            lambda row: RECORD_COLLECTION.get_signal_sample(row.record, row.start, row.end),
            axis=1
        )
        try:
            samples = np.stack(samples).astype(np.float32)
        except ValueError:
            print('Error')
            batch.to_csv('errors.csv')
        labels = batch.annot.map(LABEL_MAP).values
        
        if as_tensor:
            return batch_idx, torch.from_numpy(samples), torch.from_numpy(labels)
        else:
            return batch_idx, samples, labels
        
    def flush(self):
        self.reset()
    
    @classmethod
    def load_train_test_datasets(cls, random_seed=None, fold=1, transform=None):
        if fold=='all':
            train_dataset = None
            test_dataset = cls(
                load_expanded_split_map(),
                random_seed,
                transform
            )
        else:
            gb = load_expanded_split_map().groupby(f'split_{fold}')
            test_df = gb.get_group(True)
            train_df = gb.get_group(False)

            train_dataset = cls(
                train_df,
                random_seed,
                transform
            )
            test_dataset = cls(
                test_df,
                random_seed,
                transform
            )

        return train_dataset, test_dataset

class AfibDataset(): # Don't use, gives bad results
    def __init__(self, sample_df, window_size, set_size, random_seed, transform=None):
        self.sample_df = sample_df
        self.transform = transform
        self.window_size = window_size
        self.set_size = set_size
        
        # randomness control for consistent datasets
        self.random_seed = random_seed
        self.count = 0
        self.should_reset = False
        self.random_state = np.random.RandomState(self.random_seed)
    
    def reset(self):
        self.reset_seed = False
        self.count = 0
        self.random_state = np.random.RandomState(self.random_seed)
    
    def get_batch(self, size, as_tensor=True):
        if self.should_reset:
            self.reset()
        
        window_size = self.window_size
        def get_subsample(record, start, end):
            
            rand_start = self.random_state.choice(np.arange(start, end, 1)) # self.window_size // 2
            rand_end = rand_start + window_size
            return RECORD_COLLECTION.get_signal_sample(record, rand_start, rand_end)
            
        batch = self.sample_df.sample(
            size,
            replace=True,
            weights='duration',
            random_state=self.random_state)
        batch.end = batch.end - window_size + 1
        samples = batch.apply(
            lambda row: get_subsample(row.record, row.start, row.end),
            axis=1
        )
        samples = np.stack(samples).astype(np.float32)
        labels = batch.annot.map(LABEL_MAP).values
        self.count += size
        
        if self.count >= self.set_size:
            self.should_reset=True
        
        if as_tensor:
            return torch.from_numpy(samples), torch.from_numpy(labels)
        else:
            return samples, labels
        
    def flush(self):
        self.reset()
    
    @classmethod
    def load_train_test_datasets(cls, window_size, train_size, test_size, random_seed, fold=1, transform=None):
        gb = load_split_map().groupby(f'split_{fold}')
        test_df = gb.get_group(True)
        train_df = gb.get_group(False)

        train_dataset = cls(
            train_df,
            window_size,
            train_size,
            random_seed,
            transform
        )
        test_dataset = cls(
            test_df,
            window_size,
            test_size,
            random_seed,
            transform
        )

        return train_dataset, test_dataset