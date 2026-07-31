import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

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
    """
    Configuration class holding all paths and parameters required for the inference pipeline.
    """
    train_soundscapes = settings['train_soundscapes']
    test_soundscapes = settings['train_soundscapes']
    submission_csv = settings['submission_csv']
    taxonomy_csv = settings['taxonomy_csv']
    device = 'cuda' 
    small_labels = [
        '1139490', '1192948', '1194042', '126247', '1346504', '134933', '135045', '1462711', '1462737', 
        '1564122', '21038', '21116', '22333', '22976', '24272', '24292', '24322', '41778', '41970',
        '42007', '42087', '42113', '46010', '47067', '476537', '476538', '48124', '50186', '523060', 
        '528041', '52884', '548639', '555086', '555142', '566513', '64862', '65336', '65344', '65349', 
        '65373', '65419', '65547', '65962', '66016', '66531', '66578', '66893', '67082', '67252', '714022', 
        '715170', '787625', '81930', '868458', '963335', 'ampkin1', 'bafibi1', 'blctit1', 'bobher1',
        'bubcur1', 'cocher1', 'grysee1', 'norscr1', 'olipic1', 'piwtyr1', 'plctan1', 'plukit1', 'rosspo1', 
        'royfly1', 'rutpuf1', 'sahpar1', 'shghum1', 'tbsfin1', 'turvul', 'whmtyr1', 'woosto']
    debug_count = 4

model_cfgs = []
########################################################################################
class MyCFG(CFG):
    model_settings = [
        {
            'path': f"{settings['data_cache_dir']}/efficientnet_b0_semi.pth",
            'name': 'efficientnet_b0', 'weight':1.0,
        },
        {
            'path': f"{settings['data_cache_dir']}/efficientnet_b1_semi.pth",
            'name': 'efficientnet_b1', 'weight':1.1,
        },
        {
            'path': f"{settings['data_cache_dir']}/efficientnet_b2_semi.pth",
            'name': 'efficientnet_b2', 'weight':1.2,
        },
        {
            'path': f"{settings['data_cache_dir']}/efficientnet_b3_semi.pth",
            'name': 'efficientnet_b3', 'weight':1.3,
        },
        {
            'path': f"{settings['data_cache_dir']}/efficientnet_b4_semi.pth",
            'name': 'efficientnet_b4', 'weight':1.4,
        },

        {
            'path': f"{settings['data_cache_dir']}/tf_efficientnetv2_b0_semi.pth",
            'name': 'tf_efficientnetv2_b0', 'weight':1.0,
        },
        {
            'path': f"{settings['data_cache_dir']}/tf_efficientnetv2_b1_semi.pth",
            'name': 'tf_efficientnetv2_b1', 'weight':1.1,
        },
        {
            'path': f"{settings['data_cache_dir']}/tf_efficientnetv2_b2_semi.pth",
            'name': 'tf_efficientnetv2_b2', 'weight':1.2,
        },
        {
            'path': f"{settings['data_cache_dir']}/tf_efficientnetv2_b3_semi.pth",
            'name': 'tf_efficientnetv2_b3', 'weight':1.3,
        },
        {
            'path': f"{settings['data_cache_dir']}/tf_efficientnetv2_s_semi.pth",
            'name': 'tf_efficientnetv2_s', 'weight':1.5,
        },
    ]
    
    in_channels = 1
    batch_size = 16
    weight = 1.0
    use_tta = False  
    tta_count = 3   

    sed_threshold = 0.5
    sed_reduce = 'max'
    
    FS = 32000
    WINDOW_SIZE  = 10
    N_TARGETS = 1
    TEST_DURATION = 10
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
    POWER = 2.0
    N_MELFOLDS = 1
    FOLD_TYPE = 'fold'
    IS_CROSS_CUT = False
model_cfgs.append(MyCFG())

########################################################################################

for i, _cfg in enumerate(model_cfgs):
    _cfg.output_i = i

class SmoothAUCLoss():
    pass

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

def process_audio_file(audio_data, cfg):
    """Process a single audio file to get the mel spectrogram"""
    #audio_data, _ = librosa.load(audio_path, sr=cfg.FS)    
    mel_spec = audio2melspec(audio_data, cfg)
    
    # Resize spectrogram to the target shape if necessary.
    piece_len = int(5*cfg.FS)
    target_shape = ((len(audio_data)*cfg.TARGET_SHAPE[0])//piece_len, cfg.TARGET_SHAPE[1])
    mel_spec = cv2.resize(mel_spec, target_shape, interpolation=cv2.INTER_LINEAR)

    mel_piece_len = (cfg.TARGET_SHAPE[0]*cfg.TEST_DURATION) // 5
    mel_specs = []
    for starti in range(0, mel_spec.shape[-1], mel_piece_len):
        mel_spec_norm = librosa.power_to_db(mel_spec[:, starti:starti+mel_piece_len], ref=1.0)
        #mel_spec_norm = (mel_spec_norm - mel_spec_norm.min()) / (mel_spec_norm.max() - mel_spec_norm.min() + 1e-8)
        mel_specs.append(mel_spec_norm.astype(np.float32))
    #assert len(mel_specs) == len(audio_data)/piece_len
    return mel_specs

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

def pad_framewise_output(framewise_output, frames_num):
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output

class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, cfg):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(cfg.TARGET_SHAPE[1])

        base_model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=cfg.in_channels,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlock(
            in_features, cfg.num_classes, activation="linear")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        x = self.encoder(x)
        
        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        #x = F.dropout(x, p=cfg.dropout_rate, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        #x = F.dropout(x, p=cfg.dropout_rate, training=self.training)

        clipwise_output, norm_att, framewise_output = self.att_block(x)

        #framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, frames_num)

        return framewise_output

def get_segment_probs(frame_probs, cfg, use_interpolate=False):
    seg_len = (frame_probs.shape[-1]*cfg.WINDOW_SIZE) / cfg.TEST_DURATION
    assert seg_len == int(seg_len), seg_len
    seg_len = int(seg_len)

    frame_probs = np.concatenate(frame_probs, axis=-1)
    frame_probs = np.stack(
        [frame_probs[..., i:i+seg_len] for i in range(0, frame_probs.shape[-1], seg_len)],
        axis=0
    )

    if cfg.sed_reduce == 'mean':
        seg_probs = []
        for _probs in frame_probs:
            new_probs = []
            for _prob in _probs:
                selected_prob = _prob[_prob>cfg.sed_threshold]
                if len(selected_prob) > 0:
                    new_probs.append(selected_prob.mean())
                else:
                    new_probs.append(0)
            seg_probs.append(new_probs)
        seg_probs = np.array(seg_probs)
    elif cfg.sed_reduce == 'max':
        seg_probs = frame_probs.max(axis=-1)
    #print(seg_probs.shape, seg_probs)
    return seg_probs

class BirdCLEF2025Pipeline:
    """
    Pipeline for the BirdCLEF-2025 inference task.

    This class organizes the complete inference process:
      - Loading taxonomy data.
      - Loading and preparing the trained models.
      - Processing audio files into mel spectrograms.
      - Making predictions on each audio segment.
      - Creating the submission file.
      - Post-processing the submission to smooth predictions.
    """
    def __init__(self, cfg):
        """
        Initialize the inference pipeline with the given configuration.
        
        :param cfg: Configuration object with paths and parameters.
        """
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
        self.small_label_mask = np.array([l in self.cfg.small_labels for l in self.species_ids], dtype=bool)
        self.large_label_mask = np.array([not l in self.cfg.small_labels for l in self.species_ids], dtype=bool)
        print(f"Number of classes: {len(self.species_ids)}")

    def load_models(self):
        """
        Load all found model files and prepare them for ensemble inference.
        
        :return: List of loaded PyTorch models.
        """
        self.models = []
        # Load each model file.
        for model_settings in self.cfg.model_settings:
            model_name = model_settings['name']
            model_path = model_settings['path']
            print(f"Loading model: {model_path}")
            self.cfg.num_classes = len(self.species_ids)
            
            checkpoint = torch.load(model_path, map_location=torch.device(self.cfg.device))
            model = BirdCLEFModel(model_name, self.cfg)
            model = AveragedModel(model)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.module
            model.eval()       
            #model = torch_to_ov(model, is_tf=True)
            model = model.to(self.cfg.device)
            model_settings['model'] = model
            self.models.append(model_settings)
        
        return self.models

    def apply_tta(self, spec, tta_idx):
        """
        Apply test-time augmentation (TTA) to the spectrogram.
        
        :param spec: Input mel spectrogram.
        :param tta_idx: Index indicating which TTA to apply.
        :return: Augmented spectrogram.
        """
        if tta_idx == 0:
            # No augmentation.
            return spec
        elif tta_idx == 1:
            # Time shift (horizontal flip).
            return np.flip(spec, axis=1)
        elif tta_idx == 2:
            # Frequency shift (vertical flip).
            return np.flip(spec, axis=0)
        else:
            return spec

    def predict_on_spectrogram(self, audio_path):
        """
        Process a single audio file and predict species presence for each 5-second segment.
        
        :param audio_path: Path to the audio file.
        :return: Tuple (row_ids, predictions) for each segment.
        """
        predictions = []
        row_ids = []
        soundscape_id = Path(audio_path).stem
        
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=self.cfg.FS)
        if self.cfg.N_TARGETS == 2:
            offset = self.cfg.WINDOW_SIZE * self.cfg.FS // 2
            audio_data1 = np.concatenate([audio_data[:offset], audio_data[:-offset]], axis=0) 
            audio_data2 = np.concatenate([audio_data[offset:], audio_data[-offset:]], axis=0)
            audio_data =  audio_data1 + audio_data2  
        #assert len(audio_data) == self.cfg.TEST_DURATION*self.cfg.FS
        
        import multiprocessing as mp

        mel_spec = process_audio_file(audio_data, self.cfg)
        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(1)
        mel_spec_tensor = mel_spec_tensor.to(self.cfg.device)

        segment_preds = 0
        inter_weights = 0
        for model_settings in self.models:
            with torch.no_grad():
                outputs = model_settings['model'](mel_spec_tensor).detach().cpu().numpy()
                probs = 1 / (1 + np.exp(-outputs))
                probs = get_segment_probs(probs, self.cfg)   
                segment_preds += probs * model_settings['weight']
                inter_weights += model_settings['weight']
        predictions = segment_preds / inter_weights

        total_segments = int(len(audio_data) / (self.cfg.FS * self.cfg.WINDOW_SIZE))
        row_ids = []
        for segment_idx in range(total_segments):            
            end_time_sec = (segment_idx + 1) * self.cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)
        
        return row_ids, predictions
            
    def run_inference(self):
        """
        Run inference on all test soundscape audio files.
        
        :return: Tuple (all_row_ids, all_predictions) aggregated from all files.
        """
        test_files = list(Path(self.cfg.test_soundscapes).glob('*.ogg'))
        if len(test_files) == 0:
            test_files = list(Path(self.cfg.train_soundscapes).glob('*.ogg'))
            print(f"Debug mode enabled, using only {self.cfg.debug_count} files")
            test_files = test_files[:self.cfg.debug_count]
        
        test_files = sorted(test_files)        
        print(f"Found {len(test_files)} test soundscapes")

        all_row_ids = []
        all_predictions = []

        for audio_path in test_files:
            row_ids, predictions = self.predict_on_spectrogram(str(audio_path))
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
        
        return all_row_ids, all_predictions

    def create_submission(self, row_ids, predictions):
        """
        Create the submission dataframe based on predictions.
        
        :param row_ids: List of row identifiers for each segment.
        :param predictions: List of prediction arrays.
        :return: A pandas DataFrame formatted for submission.
        """
        print("Creating submission dataframe...")
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(self.species_ids):
            submission_dict[species] = [pred[i] for pred in predictions]

        submission_df = pd.DataFrame(submission_dict)
        submission_df.set_index('row_id', inplace=True)

        sample_sub = pd.read_csv(self.cfg.submission_csv, index_col='row_id')
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: Missing {len(missing_cols)} species columns in submission")
            for col in missing_cols:
                submission_df[col] = 0.0

        submission_df = submission_df[sample_sub.columns]
        submission_df = submission_df.reset_index()
        
        return submission_df

    def run(self):
        """
        Main method to execute the complete inference pipeline.
        
        This method:
          - Loads the pre-trained models.
          - Processes test audio files and runs predictions.
          - Creates the submission CSV.
          - Applies smoothing to the predictions.
        """
        start_time = time.time()
        print("Starting BirdCLEF-2025 inference...")
        print(f"TTA enabled: {self.cfg.use_tta} (variations: {self.cfg.tta_count if self.cfg.use_tta else 0})")
        
        self.load_models()
        if not self.models:
            print("No models found! Please check model paths.")
            return
        
        print(f"Model usage: {'Single model' if len(self.models) == 1 else f'Ensemble of {len(self.models)} models'}")
        row_ids, predictions = self.run_inference()
        submission_df = self.create_submission(row_ids, predictions)
        submission_df.to_csv(f"{settings['data_cache_dir']}/semi_submission_{self.cfg.output_i}.csv", index=False)
        
        end_time = time.time()
        print(f"Inference completed in {(end_time - start_time) / 60:.2f} minutes")

        del self.models
        gc.collect()

def smooth_submission(sub, smooth_factor=0.15):
    """
    Post-process the submission CSV by smoothing predictions to enforce temporal consistency.
    
    For each soundscape (grouped by the file name part of 'row_id'), each row's predictions
    are averaged with those of its neighbors using defined weights.
    
    :param submission_path: Path to the submission CSV file.
    """
    print("Smoothing submission predictions...")
    rev_smooth_factor = 1 - smooth_factor
    rev_smooth_factor2 = 1 - (smooth_factor*2)
    
    cols = sub.columns[1:]
    # Extract group names by splitting row_id on the last underscore
    groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        # Get indices for the current group
        idx = np.where(groups == group)[0]
        sub_group = sub.iloc[idx].copy()
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()
        
        if predictions.shape[0] > 1:
            # Smooth the predictions using neighboring segments
            new_predictions[0] = (predictions[0]*rev_smooth_factor) + (predictions[1]*smooth_factor)
            new_predictions[-1] = (predictions[-1]*rev_smooth_factor) + (predictions[-2]*smooth_factor)
            for i in range(1, predictions.shape[0]-1):
                new_predictions[i] = (predictions[i-1]*smooth_factor) + (predictions[i]*rev_smooth_factor2) + (predictions[i+1]*smooth_factor)
        # Replace the smoothed values in the submission dataframe
        sub.iloc[idx, 1:] = new_predictions
    return sub

def save_final_submission(model_cfgs=model_cfgs):
    preds = 0
    weight_sum = 0
    for cfg in model_cfgs:
        sub = pd.read_csv(f"{settings['data_cache_dir']}/semi_submission_{cfg.output_i}.csv")
        preds += sub.iloc[:, 1:].values * cfg.weight
        weight_sum += cfg.weight
    preds /= weight_sum
    sub.iloc[:, 1:] = preds
    
    sub = smooth_submission(sub)
    sub.to_csv(f"{settings['data_cache_dir']}/semi_train_preds.csv", index=False)
    print(f"Smoothed submission saved")

# Run the BirdCLEF2025 Pipeline:
if __name__ == "__main__":
    for cfg in model_cfgs:
        pipeline = BirdCLEF2025Pipeline(cfg)
        pipeline.run()
        del pipeline

    save_final_submission()