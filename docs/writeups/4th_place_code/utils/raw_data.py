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

from tqdm.auto import tqdm
import timm
import soundfile as sf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import settings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    seed = 42
    train_datadir = settings['train_datadir']
    train_csv = settings['train_csv']
    train_voice_data = settings['train_voice_data']

    output_dir = settings['data_cache_dir'] + '/raw_data'
    taxonomy_csv = settings['taxonomy_csv']

    USE_AUGMENT = False
    FS = 32000
    TARGET_DURATION = 10.0
    MIN_SOUND_DURATION= 5.0
    
cfg = CFG()

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
        audio_data = new_audio_data[:piece_len]
    else:
        audio_data = audio_data[:piece_len]

    return audio_data.astype(np.float32)

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
        purename = filepath.split('/')[-1].split('.')[0]
        
        audio = process_audio_file(filepath, cfg)
        sf.write(f'{cfg.output_dir}/{purename}.flac', audio, cfg.FS)
            
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