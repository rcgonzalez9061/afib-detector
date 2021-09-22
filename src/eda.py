import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wfdb_ext import Record
import wfdb
from PIL import Image, ImageDraw, ImageFont
from functools import partial
from scipy import fft
import glob, os
from utils import load_project_config

PROJECT_CONFIG = load_project_config()
PROJECT_DIR = PROJECT_CONFIG['project_dir']

SAMPLING_RATE = 250
LABEL_MAP_PATH = os.path.join(PROJECT_DIR, "data", "cleaned", "label_map.csv")
SIGNAL_2x2_OUTPATH = os.path.join(PROJECT_DIR, "assets", "2x2.png")
SIGNAL_1x4_OUTPATH = os.path.join(PROJECT_DIR, "assets", "1x4.png")
DFT_2x2_OUTPATH = os.path.join(PROJECT_DIR, "assets", "2x2_dft.png")
DFT_1x4_OUTPATH = os.path.join(PROJECT_DIR, "assets", "1x4_dft.png")
MEAN_ECG_OUTPATH = os.path.join(PROJECT_DIR, "assets", "mean_ecg_dft.png")


def load_label_map():
    return pd.read_csv(LABEL_MAP_PATH, dtype={"record": str})

def generate_label_maps():
    dat_pattern = os.path.join(PROJECT_DIR, 'data', 'physionet', 'afdb', '*.dat')
    record_paths = pd.Series(glob.glob(dat_pattern))
    records = record_paths.str.extract("(\d+\.dat)", expand=False).str.replace(
        "\.dat", ""
    )
    records_df = pd.DataFrame(
        data={"record": records, "path": record_paths.str.replace("\.dat", "")}
    ).set_index("record")

    label_maps = []
    for record_name, path in records_df.path.iteritems():
        label_map_df = Record(path).label_map()
        label_map_df["record"] = record_name
        label_maps.append(label_map_df)

    label_maps_df = pd.concat(label_maps)
    label_maps_df.annot = label_maps_df.annot.str.replace("^\(", "")
    label_maps_df.to_csv(LABEL_MAP_PATH, index=False)
    return label_maps_df


def generate_grouped_label_table():
    label_maps_df = generate_label_maps()
    label_maps_df = label_maps_df.rename({'annot': 'Label'}, axis=1)
    label_maps_gb = label_maps_df.groupby("Label")
    total_durations = label_maps_gb.duration.sum() / SAMPLING_RATE / 60
    total_durations.name = "Total Duration<br>(Minutes)"
    total_durations = total_durations.to_frame()
    total_durations["Total<br>Duration (%)"] = (
        total_durations["Total Duration<br>(Minutes)"]
        / total_durations["Total Duration<br>(Minutes)"].sum()
    )
    total_durations["Unique<br>Occasions"] = label_maps_gb.record.count()
    total_durations["Min<br>Duration"] = label_maps_gb.duration.min()
    total_durations["Avg Duration<br>(Samples)"] = label_maps_gb.duration.mean().astype(int)
    total_durations["Long Samples<br>(>30s)"] = (
        (label_maps_df.duration > SAMPLING_RATE * 30).groupby(label_maps_df.Label).sum()
    )
    total_durations = (
        total_durations
        .style.format({
            "Total Duration<br>(Minutes)": "{:,.2f}".format,
            "Total<br>Duration (%)": "{:,.2%}".format,
            "Avg Duration": "{:,.2f}".format,
            "Avg Duration<br>(Samples)": "{:,}".format,
        })
    )
    return total_durations


def pick_random_subsample(start, end, duration):
    valid_end = end - duration + 1
    sample_start = np.random.choice(np.arange(start, valid_end))
    sample_end = sample_start + duration
    return sample_start, sample_end


def random_sample_record(record_path, start, end, duration):
    sample_start, sample_end = pick_random_subsample(start, end, duration)
    return Record(record_path, sampfrom=sample_start, sampto=sample_end)


def sample_signals(sample_length, random_seed, num_samples=10, with_figs=True):
    if random_seed is not None:
        np.random.seed(random_seed)

    label_maps_df = load_label_map()
    label_maps_df = label_maps_df[label_maps_df.duration >= sample_length]
    label_maps_df['start_end'] = [tup for tup in zip(label_maps_df.start, label_maps_df.end)]
    signal_sample_grouped = label_maps_df.groupby("annot")

    signal_sample = signal_sample_grouped.record.unique().apply(lambda seq: np.random.choice(seq, size=num_samples)).explode().to_frame().reset_index()
    signal_sample = pd.concat([label_maps_df[(label_maps_df.record==row.record) & (label_maps_df.annot==row.annot)].sample() for idx, row in signal_sample.iterrows()]).reset_index(drop=True)
    
    # randomly clip chosen signals
    rand_start_indices = pd.Series([tup for tup in zip(signal_sample.start, signal_sample.end-sample_length)]).apply(lambda tup: np.random.randint(tup[0], tup[1]))
    signal_sample.start = rand_start_indices
    signal_sample.end = rand_start_indices + sample_length
    
    sample_records = signal_sample.apply(
        lambda row: Record(
            "data/physionet/afdb/" + row.record, row.start, row.end
        ),
        axis=1,
    )
    signal_sample["p_signal"] = sample_records.apply(lambda record: record.p_signal)
    if with_figs:
        signal_sample["plot"] = sample_records.apply(
            lambda record: record.plot(
                annotations=False, time_units="seconds", figsize=(10, 3), return_fig=True
            )
        )
        signal_sample["dft_plot"] = sample_records.apply(
            lambda record: record.plot_dft(
                figsize=(10, 3), return_fig=True
            )
        )
    return signal_sample


def fig_to_image(fig):
    fig.canvas.draw()
    return Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def fig_to_image_hdd(fig, idx=0):
    path = f"data/temp/image{idx}.png"
    fig.savefig(path, bbox_inches="tight")
    return Image.open(path)


def stack_figs(figs, title=None):
    "Vertically stack a sequence of figures. Returns a PIL Image of stack figures."
    images = [fig_to_image_hdd(fig, idx) for idx, fig in enumerate(figs)]
    width = max([img.size[0] for img in images])
    total_height = sum([image.size[1] for image in images])

    new_image = images[0].copy()
    new_image_draw = ImageDraw.Draw(new_image)
    new_image_draw.rectangle([(0, 0), new_image.size], fill="white")
    top = 0

    if title is not None:  # draw title
        font = ImageFont.truetype("Arial.ttf", size=16,)
        text_w, text_h = new_image_draw.textsize(title, font=font)
        total_height += text_h
        top += text_h
        new_image = new_image.resize(size=(width, total_height))
        new_image_draw = ImageDraw.Draw(new_image)
        new_image_draw.text(((width - text_w) / 2, 0), title, fill="black", font=font)
    else:
        new_image = new_image.resize(size=(width, total_height))

    for idx, img in enumerate(images):
        img_width, img_height = img.size
        new_image.paste(img, (width - img_width, top))
        top += img.size[1]

    return new_image


def stack_imgs_2x2(imgs):
    if len(imgs) != 4:
        raise ValueError("arg `imgs` must contain 4 images.")

    sizes = [img.size for img in imgs]
    widths = [size[0] for size in sizes]
    max_width = max(widths)

    # copy and resize image (necessary to preserve resolution)
    new_img = imgs[0].copy()
    width, height = new_img.size
    new_size = (max_width * 2, height * 2)
    new_img = new_img.resize(new_size)

    # fill background with white
    new_img_draw = ImageDraw.Draw(new_img)
    new_img_draw.rectangle([(0, 0), new_img.size], fill="white")

    # extract widths and heights
    img0_width, img0_height = imgs[0].size
    img1_width, img1_height = imgs[1].size
    img2_width, img2_height = imgs[2].size
    img3_width, img3_height = imgs[3].size

    # insert images
    new_img.paste(imgs[0], (max_width - img0_width, 0))
    new_img.paste(imgs[1], (max_width * 2 - img1_width, 0))
    new_img.paste(imgs[2], (max_width - img2_width, img0_height))
    new_img.paste(imgs[3], (max_width * 2 - img3_width, img0_height))

    return new_img

def hstack_imgs(imgs):
    sizes = [img.size for img in imgs]
    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    total_width = sum(widths)
    
    # copy and resize image (necessary to preserve resolution)
    new_img = imgs[0].copy()
    width, height = new_img.size
    new_size = (total_width, max(heights))
    new_img = new_img.resize(new_size)

    # fill background with white
    new_img_draw = ImageDraw.Draw(new_img)
    new_img_draw.rectangle([(0, 0), new_img.size], fill="white")
    
    # paste images sequentially
    right = 0
    for img in imgs:
        new_img.paste(img, (right, 0))
        right += img.size[0]
        
    return new_img

def generate_quad_plots(random_seed=None):
    sample_length = SAMPLING_RATE * 5
    signal_sample = sample_signals(sample_length, random_seed, num_samples=4)
    
    # 2x2 signal plot 
    grouped_plots = signal_sample.groupby("annot")["plot"].apply(list)
    grouped_plots = [
        stack_figs(figs, label) for label, figs in grouped_plots.iteritems()
    ]
    stack_imgs_2x2(list(grouped_plots)).save(SIGNAL_2x2_OUTPATH)
    hstack_imgs(list(grouped_plots)).save(SIGNAL_1x4_OUTPATH)
    
    # 2x2 DFT plot 
    grouped_plots = signal_sample.groupby("annot")["dft_plot"].apply(list)
    grouped_plots = [
        stack_figs(figs, label) for label, figs in grouped_plots.iteritems()
    ]
    stack_imgs_2x2(list(grouped_plots)).save(DFT_2x2_OUTPATH)
    hstack_imgs(list(grouped_plots)).save(DFT_1x4_OUTPATH)
    
def generate_combined_mean_dft_fig(sig_len, random_seed, samples_per_label):
    sample = sample_signals(sig_len, random_seed=random_seed, num_samples=samples_per_label, with_figs=False)
    sample['p_signal_dft'] = sample.p_signal.apply(partial(fft.rfft, axis=0, norm='ortho'))
    sample['p_signal_dft'] = sample['p_signal_dft'].abs()
    
    p_signal_dft_grouped = sample.groupby('annot').p_signal_dft
    ECG1_DFT_mean = p_signal_dft_grouped.apply(lambda seq: np.vstack(seq.apply(lambda arr: arr[:,0])).mean(axis=0))
    ECG2_DFT_mean = p_signal_dft_grouped.apply(lambda seq: np.vstack(seq.apply(lambda arr: arr[:,1])).mean(axis=0))
    max_value = max((
        ECG1_DFT_mean.apply(np.max).max(),
        ECG2_DFT_mean.apply(np.max).max()
    ))
    
    ecg1_fig = fig_to_image_hdd(generate_mean_dft_fig(ECG1_DFT_mean, sig_len, SAMPLING_RATE, title='ECG1', figsize=(10,5), ylabel=True, max_value=max_value), idx=0)
    ecg2_fig = fig_to_image_hdd(generate_mean_dft_fig(ECG2_DFT_mean, sig_len, SAMPLING_RATE, title='ECG2', figsize=(10,5), ylabel=False, max_value=max_value), idx=1)
    
    hstack_imgs((ecg1_fig, ecg2_fig)).save(MEAN_ECG_OUTPATH)
    

def generate_mean_dft_fig(dft_mean_seq, sig_len, fs, title=None, figsize=None, ylabel=True, max_value=None):
    x_axis = fft.rfftfreq(sig_len, 1./fs)
    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=figsize)
    fig.suptitle(title)
    if max_value is None:
        max_value = dft_mean_seq.apply(np.max).max()

    for ax, ecg_tup in zip(axes, dft_mean_seq.items()):
        label, arr = ecg_tup

        ax.set_xlim(0, arr.size)
        ax.imshow(arr[np.newaxis,:] / max_value, cmap="plasma", aspect="auto")
        if ylabel:
            ax.set_ylabel(label)
        ax.set_yticks([])
        if ax == axes[-1]:
            x_tick_map = {value: freq for value, freq in zip(np.arange(0, arr.size+1, 1), x_axis)}
            ax.set_xticks([val for val in ax.get_xticks() if val < arr.size])
            ax.set_xticklabels([int(x_tick_map[value]) for value in ax.get_xticks()])
            ax.set_xlabel('Freq. (Hz)')
            
    return fig

def generate_all_plots():
    generate_quad_plots(49) # random seed 49 gives good variation samples
    generate_combined_mean_dft_fig(250*5, random_seed=42, samples_per_label=200)