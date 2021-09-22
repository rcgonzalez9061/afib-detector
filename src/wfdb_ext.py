"""
An extension of the MIT wfdb library for added utility.
"""

import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy import fft
from utils import plot_ecg_dft, plot_spectrogram
import glob
import os


class Record(wfdb.io.record.Record):
    def __init__(self, record_name, sampfrom=0, sampto=None):
        record_dict = wfdb.rdrecord(record_name, sampfrom, sampto).__dict__
        super().__init__(**record_dict)
        self.record_path = record_name
        self.annotation = Annotation(record_name, sampfrom, sampto)
        self.start = sampfrom
        self.end = self.sig_len if sampto is None else sampto

    def subsample(self, sampfrom=0, sampto=None):
        return Record(self.record_path, sampfrom, sampto)

    def signal_subsample(self, sampfrom=0, sampto=None):
        return self.p_signal[sampfrom:sampto]

    def plot(
        self,
        annotations=True,
        title=None,
        time_units="samples",
        figsize=None,
        return_fig=False,
    ):
        return wfdb.plot_wfdb(
            record=self,
            annotation=self.annotation if annotations else None,
            title=title,
            time_units=time_units,
            figsize=figsize,
            return_fig=return_fig,
        )
    
    def plot_dft(
        self,
        title=None,
        figsize=(6,4),
        return_fig=False,
    ):
        return plot_ecg_dft(self.p_signal, self.fs, figsize, suptitle=f'Record: {self.record_name}')
    
    def plot_spectrograms(self, figsize=(6,4)):
        ecg1, ecg2 = self.p_signal.T
        
        fig, axes = plt.subplots(2, figsize=figsize)
        plot_spectrogram(ecg1, 250, ax=axes[0], f_cutoff=60.)
        plot_spectrogram(ecg2, 250, xlabel='Time', ax=axes[1], f_cutoff=60.)
        fig.suptitle(f'Record: {self.record_name}')
        
        

    def label_map(self):
        return self.annotation.label_map(sample_end=self.end)

    def group_label_map(self):
        return (
            self.label_map()
            .groupby("annot")
            .apply(lambda group: np.vstack((group.start, group.end)).T)
        )


class Annotation(wfdb.io.annotation.Annotation):
    def __init__(self, record_name, sampfrom=0, sampto=None):
        annot_dict = wfdb.rdann(
            record_name, "atr", sampfrom=sampfrom, sampto=sampto
        ).__dict__
        del annot_dict["ann_len"]
        super().__init__(**annot_dict)

    def label_map(self, sample_end=-1):
        label_map_df = pd.DataFrame(
            {
                "annot": self.aux_note,
                "start": self.sample,
                "end": np.roll(self.sample, -1) - 1,
            }
        )
        label_map_df.iloc[-1, -1] = sample_end
        label_map_df["duration"] = label_map_df.end - label_map_df.start
        return label_map_df


class RecordCollection():
    def __init__(self, data_folder):
        record_pattern = os.path.join(data_folder, '*.dat')
        record_paths = pd.Series(glob.glob(record_pattern)).str.slice(stop=-4)
        records = record_paths.str.slice(start=-5)
        self.records_map = {record: Record(record_path) for record, record_path in zip(list(records), list(record_paths))}
        
    def get_record(self, record):
        return self.records_map[record]
    
    def get_signal_sample(self, record, sampfrom=0, sampto=None):
        return self.get_record(record).signal_subsample(sampfrom, sampto).T
