import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path
import soundfile as sf

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
import timm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import settings

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    train_soundscapes = settings['train_soundscapes']
    taxonomy_csv = settings['taxonomy_csv']
    output_dir = settings['data_cache_dir'] + '/raw_semi_data'

model_cfgs = []
########################################################################################
class MyCFG(CFG):
    FS = 32000
    WINDOW_SIZE  = 10
    TEST_DURATION = 60
model_cfgs.append(MyCFG())

########################################################################################

for i, _cfg in enumerate(model_cfgs):
    _cfg.output_i = i

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
        power=cfg.POWER
    )
    mel_spec_norm = mel_spec  
    return mel_spec_norm


class BirdCLEF2025Pipeline:
    def __init__(self, cfg):
 
        self.cfg = cfg
        self.taxonomy_df = None
        self.species_ids = []
        self.models = []
        self._load_taxonomy()

    def _load_taxonomy(self):
        """
        Load taxonomy data from CSV and extract species identifiers.
        """
        print("Loading taxonomy data...")
        self.taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = self.taxonomy_df['primary_label'].tolist()
        print(f"Number of classes: {len(self.species_ids)}")


    def find_model_files(self):
        """
        Find all .pth model files in the specified model directory.
        
        :return: List of model file paths.
        """
        model_files = []
        model_dir = Path(self.cfg.model_path)
        for path in model_dir.glob('**/*'):
            model_files.append(str(path))
        model_files = sorted(model_files)
        return model_files

    def predict_on_spectrogram(self, audio_path):
        """
        Process a single audio file and predict species presence for each 5-second segment.
        
        :param audio_path: Path to the audio file.
        :return: Tuple (row_ids, predictions) for each segment.
        """
        mel_specs = []
        row_ids = []
        soundscape_id = Path(audio_path).stem
        
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=self.cfg.FS)
        assert len(audio_data) == self.cfg.TEST_DURATION*self.cfg.FS
    
        segment_len = (self.cfg.FS*self.cfg.WINDOW_SIZE)
        assert audio_data.shape[-1]//segment_len == 6, [audio_data.shape[-1], segment_len]
        for starti in range(0, audio_data.shape[-1], segment_len):
            segment_idx = starti // segment_len
            end_time_sec = (segment_idx + 1) * self.cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            while True:
                try:
                    sf.write(f'{cfg.output_dir}/{row_id}.flac', audio_data[starti:starti+segment_len], cfg.FS)
                    break
                except:
                    print(f'{row_id}.flac failed to save, try again...')
            
    def run_inference(self):
        """
        Run inference on all test soundscape audio files.
        
        :return: Tuple (all_row_ids, all_predictions) aggregated from all files.
        """
        test_files = list(Path(self.cfg.train_soundscapes).glob('*.ogg'))
        
        test_files = sorted(test_files)
        print(f"Found {len(test_files)} test soundscapes")

        all_row_ids = []
        all_predictions = []

        for audio_path in test_files:
            self.predict_on_spectrogram(str(audio_path))
        
        return all_row_ids, all_predictions

    def run(self):
        row_ids, predictions = self.run_inference()
        gc.collect()

# Run the BirdCLEF2025 Pipeline:
if __name__ == "__main__":
    for cfg in model_cfgs:
        if not os.path.exists(cfg.output_dir):
            os.mkdir(cfg.output_dir)

        pipeline = BirdCLEF2025Pipeline(cfg)
        pipeline.run()
        del pipeline