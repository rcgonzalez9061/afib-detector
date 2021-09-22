import pandas as pd
import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt
import os
import json
import nbformat
from nbconvert import HTMLExporter, PDFExporter
from pathlib import Path
import torch
import pickle
from pathlib import Path
import glob

def load_project_config():
    project_dir = Path(os.path.realpath(__file__)).parent.parent
    config_path = os.path.join(project_dir, 'config.json')
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def plot_ecg_dft(p_signal, fs, figsize=None, suptitle=None):
    n_sample_fft = pd.DataFrame(np.abs(fft.rfft(p_signal, axis=0, norm='ortho')), columns=['ECG1', 'ECG2'])
    n_sample_fft.index = fft.rfftfreq(p_signal.shape[0], d=1./fs)
    n_sample_fft = n_sample_fft[n_sample_fft.index>0]
    
    fig, axes = plt.subplots(2, figsize=figsize)
    plot_dft(n_sample_fft.ECG1, ax=axes[0])
    plot_dft(n_sample_fft.ECG2, x_label='Freq. (Hz)', ax=axes[1])
    fig.suptitle(suptitle)
    return fig

def plot_dft(fdt_col, x_label=None, ax=None):
    ax = fdt_col.plot(ax=ax, ylabel=fdt_col.name, xlabel=x_label)
    
def plot_spectrogram(x, fs, xlabel=None, ylabel=None, ax=None, f_cutoff=None):
    if ax is None:
        ax = plt.axes()
    
    nperseg = 64
    f, t, Sxx = signal.spectrogram(x, fs, window='hamming', nperseg=nperseg, noverlap=nperseg//8)
    ax.pcolormesh(t, fft.fftshift(f), fft.fftshift(np.log10(Sxx), axes=0), shading='gouraud')
    
    if f_cutoff is not None:
        ax.set_ylim(top=f_cutoff)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    return ax
    
def load_model(model_path, model_class, model_args, device=torch.device('cpu')):
    model = model_class(**model_args)
    state_dict = torch.load(model_path, device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model
    
def convert_notebook(report_in_path, report_out_path, **kwargs):
    curdir = os.path.abspath(os.getcwd())
    indir, _ = os.path.split(report_in_path)
    outdir, _ = os.path.split(report_out_path)
    os.makedirs(outdir, exist_ok=True)

    config = {
        "ExecutePreprocessor": {"enabled": True, "timeout": -1},
        "TemplateExporter": {
            "exclude_output_prompt": True,
            "exclude_input": True,
            "exclude_input_prompt": True,
        },
    }

    nb = nbformat.read(open(report_in_path), as_version=4)
    html_exporter = HTMLExporter(config=config)

    # no exectute for PDFs
    config["ExecutePreprocessor"]["enabled"] = False
    pdf_exporter = PDFExporter(config=config)

    # change dir to notebook dir, to execute notebook
    os.chdir(indir)

    body, resources = html_exporter.from_notebook_node(nb)
    pdf_body, pdf_resources = pdf_exporter.from_notebook_node(nb)

    # change back to original directory
    os.chdir(curdir)

    with open(report_out_path.replace(".pdf", ".html"), "w") as fh:
        fh.write(body)

    with open(report_out_path.replace(".html", ".pdf"), "wb") as fh:
        fh.write(pdf_body)