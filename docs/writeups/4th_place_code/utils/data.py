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

def apply_audio_augmentations(audio):
    if cfg.USE_AUGMENT:
        noise_file = np.random.choice(cfg.noise_files)
        noise_data, _ = librosa.load(noise_file, sr=cfg.FS)
        if len(noise_data) < len(audio):
            n_copy = math.ceil(len(audio)/len(noise_data))
            if n_copy > 1:
                noise_data = np.concatenate([noise_data] * n_copy)    
        starti = np.random.randint(len(noise_data)-len(audio))
        noise_data = noise_data[starti:starti+len(audio)]
        audio = audio*((np.random.rand()*1.5)+0.5) + noise_data*((np.random.rand()*1.5)+0.5)

    return audio

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
    piece_len = int(cfg.TARGET_DURATION*cfg.FS)
    
    audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
    speech_segs = []
    if audio_path in cfg.voice_data:
        speech_segs = cfg.voice_data[audio_path]

    seg_len = int(cfg.FS*cfg.TARGET_DURATION)
    min_sound_len = int(cfg.FS*cfg.MIN_SOUND_DURATION)
    mask = np.ones(audio_data.shape, dtype=int).astype(bool)
    for seg in speech_segs:
        mask[int(seg['start']*cfg.FS):int(seg['end']*cfg.FS)] = False
        
    new_audio_data = audio_data[mask]
    if len(new_audio_data) >= min_sound_len:
        audio_data = new_audio_data
    else:
        audio_data = audio_data[:piece_len]
    
    mel_spec = audio2melspec(audio_data, cfg)

    target_shape = ((len(audio_data)*cfg.TARGET_SHAPE[0])//piece_len, cfg.TARGET_SHAPE[1])
    mel_spec = cv2.resize(mel_spec, target_shape, interpolation=cv2.INTER_LINEAR)
    if len(mel_spec.shape) == 3:
       mel_spec = mel_spec.transpose([2, 0, 1])
    
    if cfg.IS_FIRST10:
        mel_spec = mel_spec[..., :cfg.TARGET_SHAPE[0]*2]
    return mel_spec.astype(np.float32)

def generate_spectrograms(df):
    set_seed(cfg.seed)
    
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in df.iterrows():
        samplename = row['samplename']
        filepath = row['filepath']
        
        mel_spec = process_audio_file(filepath, cfg)
        if mel_spec is not None:
            np.save(f'{cfg.output_dir}/{samplename}.npy', mel_spec)
            
        if i%1000 == 999:
            print(f'{i+1} files passed')

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

    #return all_bird_data
import pickle

if __name__ == "__main__":
    import time
    if not os.path.exists(cfg.output_dir):
        os.mkdir(cfg.output_dir)
    
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    with open(cfg.train_voice_data, 'rb') as f:
        cfg.voice_data = pickle.load(f)
    print(f'all data: {len(train_df)}')
    train_df['file'] = [name.split('/')[-1] for name in train_df['filename'].values]
    print(f'used data: {len(train_df)}')
    
    if 'filepath' not in train_df.columns:
        train_df['filepath'] = cfg.train_datadir + '/' + train_df.filename
    if 'samplename' not in train_df.columns:
        train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    
    generate_spectrograms(train_df)