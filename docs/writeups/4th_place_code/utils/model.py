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
from torch.optim.swa_utils import AveragedModel

import argparse
import timm
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import semi_model_cfg_dict, model_cfg_dict, settings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='')
parser.add_argument('--mode', type=str, default='pretrain')
args = parser.parse_args()

if args.mode == 'pretrain': 
    cfg = semi_model_cfg_dict[args.version]
    cfg.semi_train_csv = None
    cfg.save_path = f"{settings['data_cache_dir']}/semi_models"
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    cfg.save_path = f'{cfg.save_path}/{args.version}.pth'
else:
    assert args.mode == 'train', f'wrong training mode: {args.mode}'
    cfg = model_cfg_dict[args.version]
    cfg.save_path = f"{settings['model_dir']}/{args.version}.pth"

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

class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode

        weights = self.df['rating'].values
        weights[weights==0] = 4
        weights += 5
        self.df['weight'] = 1#(weights*len(weights)) / weights.sum()

        self.spectrograms = spectrograms
        if not self.spectrograms is None:
            for k in self.spectrograms:
                _data = np.load(self.spectrograms[k], allow_pickle=True)
                try:
                    _data = _data.item()['spec']
                except:
                    pass
                if len(_data.shape) > 2:
                    cfg.in_channels = _data.shape[0]
                break
    
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        self.samplename_id_map = {name: i for i, name in enumerate(self.df['samplename'].values)}
        self.purename_id_map = {name: i for i, name in enumerate(self.df['purename'].values)}
        self.df_samplename_dict = {samplename:[] for samplename in self.df['samplename'].values}
        for name in self.spectrograms:
            df_samplename = self.get_pure_name(name)
            if df_samplename in self.samplename_id_map:
                self.df_samplename_dict[df_samplename].append(name)
        for k in self.df_samplename_dict:
            self.df_samplename_dict[k] = sorted(self.df_samplename_dict[k])

        self.semi_samplename_dict = {}
        for ssn in self.df[self.df['primary_label']=='semi']['samplename'].values:
            base_ssn = '_'.join(ssn.split('_')[:-1])
            if base_ssn in self.semi_samplename_dict:
                self.semi_samplename_dict[base_ssn].append(ssn)
            else:
                self.semi_samplename_dict[base_ssn] = [ssn]
        #print(self.semi_samplename_dict)
        if mode == 'train':
            assert cfg.resample_primary_only
            self.init_train_dataset()
        else:
            self.used_df_samplenames = self.df['samplename'].values

        self.used_df_samplenames = [sn for sn in self.used_df_samplenames if len(self.df_samplename_dict[sn])>0]

        print(f"Found {len(self.used_df_samplenames)} matching spectrograms for {mode} dataset out of {len(self.df)} samples")

    def init_train_dataset(self):
        self.used_df_samplenames = []
        used_df = self.df[self.df['removed']==0]
        for label in self.label_to_idx:
            _samplenames = used_df[used_df['primary_label']==label]['samplename'].tolist()
            _samplenames = [sn for sn in _samplenames if len(self.df_samplename_dict[sn])>0]
            if len(_samplenames)>0 and len(_samplenames)<cfg.min_smaples_pre_class:
                np.random.shuffle(_samplenames)
                _samplenames = _samplenames * math.ceil(cfg.min_smaples_pre_class/len(_samplenames))
                self.used_df_samplenames += _samplenames[:cfg.min_smaples_pre_class]
            else:
                self.used_df_samplenames += _samplenames
        
        #for k in self.semi_samplename_dict:
        #    self.used_df_samplenames += [np.random.choice(self.semi_samplename_dict[k])]
        self.used_df_samplenames += used_df[used_df['primary_label']=='semi']['samplename'].tolist()
        np.random.shuffle(self.used_df_samplenames)
    
    def get_pure_name(self, name):             
        pure_name = name.split('__')
        if len(pure_name) > 1:
            return pure_name[0]
        name_pieces = name.split('_')
        if len(name_pieces) < 3:
            return name_pieces[0]
        return name
    
    def __len__(self):
        return len(self.used_df_samplenames)
    
    def __getitem__(self, idx): 
        samplename_cache = self.df_samplename_dict[self.used_df_samplenames[idx]]
        if self.mode == 'train':
            samplename = np.random.choice(samplename_cache)
        else:
            samplename = samplename_cache[0]

        row = self.df.iloc[self.samplename_id_map[self.get_pure_name(samplename)]]
        weight = row['weight']
        target = row[self.species_ids].astype(np.float32)
        
        filename = self.spectrograms[samplename]
        if self.mode=="train" and len(cfg.aug_spectrograms_lists)>0 and random.random()<self.cfg.preaug_prob:
            filename = np.random.choice(cfg.aug_spectrograms_lists[samplename.split('-')[-1]])
            spec = np.load(filename, allow_pickle=True)
            spec = spec.item()
            spec, noise_name = spec['spec'], spec['noise_name']
            noise_row = self.df.iloc[self.purename_id_map[noise_name]]
            noise_target = noise_row[self.species_ids].astype(np.float32)
            target = np.max(np.stack([target, noise_target], axis=0), axis=0)
        else:
            spec = np.load(filename, allow_pickle=True)
            
        spec_len = self.cfg.TRAIN_DURATION_RANGE[0]
        if self.cfg.TRAIN_DURATION_RANGE[1] > self.cfg.TRAIN_DURATION_RANGE[0]:
            spec_len = np.random.uniform(self.cfg.TRAIN_DURATION_RANGE[0], self.cfg.TRAIN_DURATION_RANGE[1])
        spec_len = (spec_len*self.cfg.data_cfg.TARGET_SHAPE[0]) // 5 
        if spec.shape[-1] < spec_len:
            pad = np.zeros([spec.shape[0], spec_len])
            pad1 = np.random.randint(spec_len-spec.shape[-1])
            pad2 = spec_len - spec.shape[-1] - pad1
            spec = np.concatenate([pad[..., :pad1], spec, pad[..., :pad2]], axis=-1)
        elif spec.shape[-1] > spec_len:
            starti = 0
            if self.mode == 'train':
                starti = np.random.randint(spec.shape[-1]-spec_len)
            spec = spec[..., starti:starti+spec_len]
        assert spec.shape[-1] == spec_len, spec.shape
        
        spec = librosa.power_to_db(spec, ref=1.0)
        #spec = (spec-spec.min()) / (spec.max()-spec.min() + 1e-8)
        
        spec = torch.tensor(spec, dtype=cfg.dtype)
        if len(spec.shape) == 2:
            spec = spec.unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)
        #print(spec.shape)
        return {
            'melspec': spec, 
            'target': torch.tensor(target, dtype=cfg.dtype),
            'weight': torch.tensor(weight, dtype=cfg.dtype),
            'filename': row['filename']
        }
    
    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""
    
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            #bias = random.uniform(-0.1, 0.1)
            spec = spec * gain #+ bias
            #spec = torch.clamp(spec, 0, 1) 
            
        return spec
    
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'weight' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    
    return result

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

def get_segment_preds(frame_preds, use_interpolate=False):
    if use_interpolate:
        target_shape = []
        return cv2.resize(frame_preds, target_shape, interpolation=cv2.INTER_LINEAR)
    #frame_preds

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(cfg.data_cfg.TARGET_SHAPE[1])

        base_model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.5
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
        #x = input_data.transpose(2,3)
        #x = torch.cat((x,x,x),1)

        #x = x.transpose(2, 3)

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

        x = F.dropout(x, p=cfg.dropout_rate, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=cfg.dropout_rate, training=self.training)

        clipwise_output, norm_att, framewise_output = self.att_block(x)

        #framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, frames_num)

        return clipwise_output, framewise_output

def get_optimizer(model, cfg):
  
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer

def get_scheduler(optimizer, cfg):
   
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None
    else:
        scheduler = None
        
    return scheduler


class AUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels == 1]
        neg_preds = preds[labels == 0]
        
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)
        
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = sample_weights[labels == 1]  # [N_pos]
            neg_weights = sample_weights[labels == 0]  # [N_neg]
        else:
            pos_weights = torch.ones_like(pos_preds) * self.pos_weight
            neg_weights = torch.ones_like(neg_preds) * self.neg_weight
        
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]
        
        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        
        return weighted_loss.mean()

class SoftAUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels>0.5]
        neg_preds = preds[labels<0.5]
        pos_labels = labels[labels>0.5]
        neg_labels = labels[labels<0.5]
        
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels-0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5-neg_labels)
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = pos_weights * sample_weights
            neg_weights = neg_weights * sample_weights
            
        
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]
        
        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        
        return weighted_loss.mean()

def get_criterion(cfg):
    criterions = []
    for criterion_name in cfg.criterions:
        if criterion_name == 'AUCLoss':
            criterion = AUCLoss()
        elif criterion_name == 'SoftAUCLoss':
            criterion = SoftAUCLoss()
        else:
            raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        criterions.append(criterion)
        
    def combined_loss(outputs, targets, weights=None, criterions=criterions):
        return sum([criterion(outputs, targets) for criterion in criterions]) / len(criterions)
    return combined_loss

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    loader.dataset.init_train_dataset()
    pbar = enumerate(loader)   
    for step, batch in pbar:
        inputs = batch['melspec'].to(device)
        targets = batch['target'].to(device)
        weights = batch['weight'].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            if cfg.mixup_rate > np.random.rand():
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                indices = torch.randperm(inputs.size(0)).to(inputs.device)
                inputs = lam * inputs + (1-lam) * inputs[indices]
                outputs, _ = model(inputs)
                
                loss = lam * criterion(outputs, targets, weights)
                loss = loss + (1-lam) * criterion(outputs, targets[indices], weights[indices])
            else:
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            
        #loss.backward()
        cfg.scaler.scale(loss).backward()
        if not cfg.max_grad_norm is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        #optimizer.step()
        cfg.scaler.step(optimizer)
        cfg.scaler.update()
        
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    aucs = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, np.mean(aucs)#, aucs

def validate(model, loader, criterion, device):
   
    model.eval()
    losses = []
    sample_losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)

            with torch.cuda.amp.autocast(enabled=cfg.apex):
                outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            cfg.scaler.scale(loss)
            sample_loss = cfg.sample_loss_func(outputs, targets).mean(dim=-1)
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())
            sample_losses.append(sample_loss.detach().cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    sample_losses = np.concatenate(sample_losses)
    aucs = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, np.mean(aucs), aucs, all_outputs, all_targets, sample_losses

def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    targets_sum = targets.sum(1)
    mask = targets_sum == targets_sum.astype(int)
    targets = targets[mask]
    outputs = outputs[mask]
    
    probs = 1 / (1 + np.exp(-outputs))
    aucs = []
    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score((targets[:, i]>0.0).astype(int), probs[:, i])
            aucs.append(class_auc)
        else:
            aucs.append(0)
    
    return aucs 

from glob import glob
import copy

def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)

    if cfg.LOAD_DATA:
        print("Loading pre-computed mel spectrograms from NPY file...")
        train_spectrograms = []
        for spectrogram_path in cfg.train_spectrogram_paths:
            for path in glob(spectrogram_path+'/*.npy'):
                train_spectrograms.append(path)
        train_spectrograms = sorted(train_spectrograms)
        np.random.shuffle(train_spectrograms)
        train_spectrograms = {path.split('/')[-1][:-4]: path for path in train_spectrograms}
        print(f"Loaded {len(train_spectrograms)} pre-computed train mel spectrograms")

        test_spectrograms = []
        for spectrogram_path in cfg.test_spectrogram_paths:
            for path in glob(spectrogram_path+'/*.npy'):
                test_spectrograms.append(path)
        test_spectrograms = sorted(test_spectrograms)
        np.random.shuffle(test_spectrograms)
        test_spectrograms = {path.split('/')[-1][:-4]: path for path in test_spectrograms}
        print(f"Loaded {len(test_spectrograms)} pre-computed test mel spectrograms")

        cfg.aug_spectrograms_lists = {}
        for spectrogram_path in cfg.aug_spectrogram_paths:
            for path in glob(spectrogram_path+'/*.npy'):
                filename = path.split('/')[-1].split('.')[0]
                if filename in cfg.aug_spectrograms_lists:
                    cfg.aug_spectrograms_lists[filename].append(path)
                else:
                    cfg.aug_spectrograms_lists[filename] = [path]        
    
    if not cfg.LOAD_DATA:
        print("Will generate spectrograms on-the-fly during training.")
        if 'filepath' not in df.columns:
            df['filepath'] = cfg.train_datadir + '/' + df.filename
        if 'samplename' not in df.columns:
            df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.fold_seed)
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
            
        print(f'\n{"="*30} Fold {fold} {"="*30}')

        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_df = df.iloc[:]
        if not cfg.extra_train_df is None:
            train_df = pd.concat([train_df, cfg.extra_train_df])
        train_df = train_df.sample(frac=1, random_state=cfg.seed*fold).reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=train_spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=test_spectrograms, mode='valid')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = BirdCLEFModel(cfg).to(cfg.dtype).to(cfg.device)
        #print(model)
        if not cfg.model_checkpoint is None:
            model_state_dict = torch.load(cfg.model_checkpoint, map_location=cfg.device)['model_state_dict']
            new_state_dict = {}
            for k in model_state_dict:
                if not 'classifier.' in k:
                    new_state_dict[k] = model_state_dict[k]
            model.load_state_dict(new_state_dict, strict=False)
            print('keep training......')
            del model_state_dict
            torch.cuda.empty_cache()
        else:
            print('trainin new model......')
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        cfg.criterion_func = criterion
        cfg.sample_loss_func = nn.BCEWithLogitsLoss(reduction='none')
        cfg.scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
        
        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        best_auc = 0
        best_epoch = 0
        auc_cache = []
        swa_model = AveragedModel(model)
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            print(f"{cfg.model_name} epoch:{epoch+1} Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            
            if epoch+1 >= cfg.acc_start_epoch:
                swa_model.update_parameters(model)
                
                val_loss, val_auc, val_aucs, preds, labels, losses = validate(model, val_loader, criterion, cfg.device)
                print(f"{cfg.model_name} epoch:{epoch+1} Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                auc_ids = np.arange(len(val_aucs))
                auc_ids = sorted(auc_ids, key=lambda x:val_aucs[x])[:20]
                low_aucs = {val_dataset.species_ids[i]:val_aucs[i] for i in auc_ids}
                print(f"20 lowest auc classes: {low_aucs}")    
               
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    print(f"{cfg.model_name} epoch:{epoch+1} New best AUC: {best_auc:.4f} at epoch {best_epoch}")
    
                    samplenames = val_dataset.used_df_samplenames
                    preds = 1 / (1+np.exp(-preds))
                    pred_dict = {'samplename':samplenames, 'loss':losses}
                    for i in range(preds.shape[-1]):
                        pred_dict[f'pred{i}'] = preds[:, i]
                    pd.DataFrame(pred_dict).to_csv(f'cv_preds{cfg.model_name}.csv', index=False)
                    
        best_scores.append(best_auc)
        print(f"\n:{cfg.model_name} epoch:{epoch+1} Best AUC for {cfg.model_name}: {best_auc:.4f} at epoch {best_epoch}")
        torch.save({'model_state_dict': swa_model.state_dict()}, cfg.save_path)

        #save_as_onnx(swa_model.module.cpu(), f"model_fold{fold}_onnx.onnx")
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)

if __name__ == "__main__":
    import time
    set_seed(cfg.seed)
    
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    cfg.num_classes = len(taxonomy_df)
    train_df['filepath'] = cfg.train_datadir + '/' + train_df.filename
    train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    train_df['purename'] = train_df.filename.map(lambda x: x.split('/')[-1].split('.')[0])
    
    pred_cache = []
    label_counts = {ln:0 for ln in taxonomy_df['primary_label'].values}
    label_to_id = {l:i for i, l in enumerate(label_counts)}
    label_ids = []
    data_label_cache = []
    label_embeddings = []
    for ri, row in train_df.iterrows():
        _labels = [row['primary_label']]
        if not cfg.cv_primary_only:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            for label in secondary_labels:
                if label in label_counts:
                    _labels.append(label)

        label_embedding = np.zeros([len(label_to_id)])
        for label in _labels:
            label_embedding[label_to_id[label]] = 1
            label_counts[label] += 1
        data_label_cache.append(_labels)
        label_embeddings.append(label_embedding)
    train_df[taxonomy_df['primary_label'].values] = label_embeddings
    train_df['removed'] = 0
    
    cv_labels = []
    for ln in label_counts:
        if label_counts[ln] >= cfg.min_smaples_for_cv:
            cv_labels.append(ln)
    cv_ids = np.zeros(len(data_label_cache)).astype(bool)
    for i, _labels in enumerate(data_label_cache):
        for label in _labels:
            if label in cv_labels:
                cv_ids[i] = True
                break

    print(f'all data: {len(train_df)}')
    cfg.extra_train_df = train_df[~cv_ids].reset_index(drop=True)
    train_df = train_df[cv_ids].reset_index(drop=True)
    
    print(f'cv data: {len(train_df)}')
    if not cfg.extra_train_csv is None:
        cfg.extra_train_df = pd.concat([cfg.extra_train_df,
                                        pd.read_csv(cfg.extra_train_csv)]).reset_index(drop=True)

    if not cfg.semi_train_csv is None:
        semi_train_df = pd.read_csv(cfg.semi_train_csv)
        semi_train_df['samplename'] = semi_train_df['row_id']
        semi_train_df['purename'] = semi_train_df['row_id']
        semi_train_df['primary_label'] = 'semi'
        semi_train_df['removed'] = 0
        for label in taxonomy_df['primary_label'].values:
            label_min = semi_train_df[label].min()
            label_max = semi_train_df[label].max()
            semi_train_df[label] = (semi_train_df[label]-label_min) / (label_max-label_min)
        label_sums = semi_train_df[taxonomy_df['primary_label'].values].values.sum(-1)
        semi_train_df.loc[label_sums<=sorted(label_sums)[int(len(label_sums)*cfg.semi_drop)], 'removed'] = 1
        del semi_train_df['row_id']
        train_df = pd.concat([train_df, semi_train_df]).reset_index(drop=True)
        
    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")
    
    run_training(train_df, cfg)
    
    print("\nTraining complete!")