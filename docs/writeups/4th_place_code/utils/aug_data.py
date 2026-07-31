import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import data_cfg_dict, settings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='')
args = parser.parse_args()

cfg = data_cfg_dict[args.version]
cfg.output_dir = f"{settings['data_cache_dir']}/{args.version}"

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(cfg.seed)

def apply_audio_augmentations(audio, audio_name):
    noise_name = audio_name
    while noise_name == audio_name:
        noise_file = np.random.choice(cfg.noise_files)
        noise_name = noise_file.split('/')[-1].split('.')[0]
    noise_data, _ = librosa.load(noise_file, sr=cfg.FS)
    if len(noise_data) < len(audio):
        n_copy = math.ceil(len(audio)/len(noise_data))
        if n_copy > 1:
            noise_data = np.concatenate([noise_data] * n_copy)
        noise_data = noise_data[:len(audio)]
    elif len(noise_data) > len(audio):
        starti = np.random.randint(len(noise_data)-len(audio))
        noise_data = noise_data[starti:starti+len(audio)]
    #print(noise_name, audio_name)
    audio = audio + noise_data
    noise_name = noise_file.split('/')[-1].split('.')[0]
    return audio, noise_name

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    #mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    #mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""

    audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
    audio_name = audio_path.split('/')[-1].split('.')[0]
    noise_name = None
    if cfg.USE_AUGMENT and len(cfg.noise_files)>0:
        audio_data, noise_name = apply_audio_augmentations(audio_data, audio_name)
    mel_spec = audio2melspec(audio_data, cfg)

    target_shape = (cfg.TARGET_SHAPE[0]*2, cfg.TARGET_SHAPE[1])
    mel_spec = cv2.resize(mel_spec, target_shape, interpolation=cv2.INTER_LINEAR)
    if len(mel_spec.shape) == 3:
       mel_spec = mel_spec.transpose([2, 0, 1])

    #mel_spec = mel_spec[..., :cfg.TARGET_SHAPE[0]*2]
    return mel_spec.astype(np.float32), audio_name, noise_name

def generate_spectrograms(audio_files):
    set_seed(cfg.seed)
    
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    for i, filepath in enumerate(audio_files):
        mel_spec, audio_name, noise_name = process_audio_file(filepath, cfg)
        filepath
        if noise_name is None:
            np.save(f'{cfg.output_dir}/{audio_name}.npy', mel_spec)
        else:
            np.save(f'{cfg.output_dir}/{audio_name}.npy', {'spec':mel_spec, 'noise_name':noise_name})
            
        if i%1000 == 999:
            print(f'{i+1} files passed')

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

    #return all_bird_data
import pickle
from glob import glob

if __name__ == "__main__":
    import time
    if not os.path.exists(cfg.output_dir):
        os.mkdir(cfg.output_dir)

    audio_files = sorted(glob(cfg.train_datadir+'/*.flac'))
    cfg.noise_files = []
    for datadir in cfg.aug_train_datadirs:
        cfg.noise_files += glob(datadir+'/*.flac')
    cfg.noise_files = sorted(cfg.noise_files)

    print('len(audio_files): ', len(audio_files))
    generate_spectrograms(audio_files)