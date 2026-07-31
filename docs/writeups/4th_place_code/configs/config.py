import json
import torch

with open('SETTINGS.json', 'r') as f:
    settings = json.load(f)

#--------------------------------------------------------- data configs ----------------------------------------------------------
data_cfg_dict = {}

class BASE_DATA_CFG:
    train_datadir = settings['train_datadir']
    train_csv = settings['train_csv']
    taxonomy_csv = settings['taxonomy_csv']

    train_voice_data = settings['train_voice_data']

    seed = 42
    num_workers = 6
    USE_AUGMENT = False
    IS_FIRST10 = True
    
    FS = 32000
    TARGET_DURATION = 5.0
    MIN_SOUND_DURATION= 5.0

class DATA_CFG(BASE_DATA_CFG):
    IS_FIRST10 =False
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_full'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_first10'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_semi_data'
    aug_train_datadirs = [
    ]
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_semi_first10'] = DATA_CFG()

#-----------------------------------------
class DATA_CFG(BASE_DATA_CFG):
    seed = 33
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_data'
    aug_train_datadirs = [
        settings['data_cache_dir'] + '/raw_data',
        settings['data_cache_dir'] + '/raw_semi_data',
    ]
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_first10_augseed33'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    seed = 55
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_data'
    aug_train_datadirs = [
        settings['data_cache_dir'] + '/raw_data',
        settings['data_cache_dir'] + '/raw_semi_data',
    ]
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_first10_augseed55'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    seed = 77
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_data'
    aug_train_datadirs = [
        settings['data_cache_dir'] + '/raw_data',
        settings['data_cache_dir'] + '/raw_semi_data',
    ]
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_first10_augseed77'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    seed = 111
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_semi_data'
    aug_train_datadirs = [
        settings['data_cache_dir'] + '/raw_data',
        settings['data_cache_dir'] + '/raw_semi_data',
    ]
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_semi_first10_augseed111'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    seed = 222
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_semi_data'
    aug_train_datadirs = [
        settings['data_cache_dir'] + '/raw_data',
        settings['data_cache_dir'] + '/raw_semi_data',
    ]
    TARGET_SHAPE = (256, 256)
    N_FFT = 2048
    HOP_LENGTH = 64
    N_MELS = 256
    FMIN = 60
    FMAX = 16000
data_cfg_dict['v1_semi_first10_augseed222'] = DATA_CFG()

#-----------------------------------------
class DATA_CFG(BASE_DATA_CFG):
    TARGET_SHAPE = (384, 192)
    N_FFT = 1536
    HOP_LENGTH = 64
    N_MELS = 192
    FMIN = 50
    FMAX = 16000
data_cfg_dict['v2_first10'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_semi_data'
    aug_train_datadirs = [
    ]
    TARGET_SHAPE = (384, 192)
    N_FFT = 1536
    HOP_LENGTH = 64
    N_MELS = 192
    FMIN = 50
    FMAX = 16000
data_cfg_dict['v2_semi_first10'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    TARGET_SHAPE = (288, 224)
    N_FFT = 2048
    HOP_LENGTH = 128
    N_MELS = 224
    FMIN = 40
    FMAX = 16000
data_cfg_dict['v3_first10'] = DATA_CFG()

class DATA_CFG(BASE_DATA_CFG):
    USE_AUGMENT = True
    train_datadir = settings['data_cache_dir'] + '/raw_semi_data'
    aug_train_datadirs = [
    ]
    TARGET_SHAPE = (288, 224)
    N_FFT = 2048
    HOP_LENGTH = 128
    N_MELS = 224
    FMIN = 40
    FMAX = 16000
data_cfg_dict['v3_semi_first10'] = DATA_CFG()


#--------------------------------------------------------- model configs ----------------------------------------------------------
semi_model_cfg_dict = {}
model_cfg_dict = {}

class BASE_MODEL_CFG:
    seed = 42
    fold_seed = 42
    num_workers = 6
    
    train_datadir = settings['train_datadir']
    train_csv = settings['train_csv']
    taxonomy_csv = settings['taxonomy_csv']
    test_soundscapes = settings['test_soundscapes']
    submission_csv = settings['submission_csv']

    semi_train_csv = f"{settings['data_cache_dir']}/semi_train_preds.csv"
    semi_drop = 0.2

    aug_spectrogram_paths = []
    LOAD_DATA = True
    extra_train_csv = None
    device = 'cuda'
    dtype = torch.float32
    n_fold = 6
    selected_folds = [0]  
    TRAIN_DURATION_RANGE = [10, 10]

    model_checkpoint = None
    pretrained = True
    in_channels = 1
  
    epochs = 20
    acc_start_epoch = 16
    batch_size = 32

    min_smaples_pre_class = 100
    resample_primary_only = True
    min_smaples_for_cv = 50
    cv_primary_only = False
    criterions = ['SoftAUCLoss']

    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5
    pct_start = 0.0
    apex = True
    max_grad_norm = None
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    preaug_prob = 0.5
    aug_prob = 0.5  
    dropout_rate = 0.5
    mixup_alpha = 0.5 
    mixup_rate = 1.0

train_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_full",
]
test_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_full",
]
data_cfg = data_cfg_dict['v1_full']
criterions = ['AUCLoss']
suffix = 'semi'

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'efficientnet_b0'
    seed = 42
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'efficientnet_b1'
    seed = 111
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'efficientnet_b2'
    seed = 121
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'efficientnet_b3'
    seed = 131
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'efficientnet_b4'
    seed = 141
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'tf_efficientnetv2_b0'
    seed = 201
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'tf_efficientnetv2_b1'
    seed = 211
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg


class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'tf_efficientnetv2_b2'
    seed = 221
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'tf_efficientnetv2_b3'
    seed = 231
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    criterions = criterions
    epochs = 60
    acc_start_epoch = 41
    model_name = 'tf_efficientnetv2_s'
    seed = 301
cfg = MODEL_CFG()
semi_model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

#---------------------------------------------------------------------------
train_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_first10",
    f"{settings['data_cache_dir']}/v1_semi_first10",
]
test_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_first10",
]
data_cfg = data_cfg_dict['v1_first10']
suffix = 'v1_first10'

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    epochs = 22
    acc_start_epoch = 18
    model_name = 'tf_efficientnet_lite0'
    seed = 2001
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    model_name = 'tf_efficientnet_lite1'
    seed = 11
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    epochs = 22
    acc_start_epoch = 18
    model_name = 'tf_efficientnet_lite2'
    seed = 2021
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnet_lite3'
    seed = 2031
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnet_lite4'
    seed = 2041
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    epochs = 23
    acc_start_epoch = 19
    model_name = 'efficientnet_b2'
    seed = 121
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnetv2_b3'
    seed = 2231
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnetv2_s'
    seed = 20211
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

#-----------------------------------------
train_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_full",
    f"{settings['data_cache_dir']}/v1_semi_first10",
]
test_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_full",
]
data_cfg = data_cfg_dict['v1_full']
suffix = 'v1_full'

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    epochs = 25
    acc_start_epoch = 21
    model_name = 'tf_efficientnetv2_b3'
    seed = 241
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnetv2_s'
    seed = 251
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

#-----------------------------------------
train_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_first10",
    f"{settings['data_cache_dir']}/v1_semi_first10",
]
test_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v1_first10",
]
aug_spectrogram_paths = [f"{settings['data_cache_dir']}/{dv}" for dv in data_cfg_dict if 'augseed' in dv]
data_cfg = data_cfg_dict['v1_first10']
suffix = 'v1_first10_aug'

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    aug_spectrogram_paths = aug_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'efficientnet_b3'
    seed = 131
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    aug_spectrogram_paths = aug_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnetv2_b2'
    seed = 221
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

#-----------------------------------------
train_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v2_first10",
    f"{settings['data_cache_dir']}/v2_semi_first10",
]
test_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v2_first10",
]
data_cfg = data_cfg_dict['v2_first10']
suffix = 'v2_first10'

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    semi_drop = 0.1
    model_name = 'tf_efficientnetv2_b3'
    seed = 72231
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    epochs = 22
    acc_start_epoch = 19
    model_name = 'efficientnet_b3'
    data_cfg = data_cfg_dict['v2_first10']
    seed = 72131
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

#-----------------------------------------
train_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v3_first10",
    f"{settings['data_cache_dir']}/v3_semi_first10",
]
test_spectrogram_paths = [
    f"{settings['data_cache_dir']}/v3_first10",
]
data_cfg = data_cfg_dict['v3_first10']
suffix = 'v3_first10'

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    model_name = 'tf_efficientnetv2_s'
    seed = 73301
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg

class MODEL_CFG(BASE_MODEL_CFG):
    train_spectrogram_paths = train_spectrogram_paths
    test_spectrogram_paths = test_spectrogram_paths
    data_cfg = data_cfg
    epochs = 20
    acc_start_epoch = 17
    model_name = 'efficientnet_b2'
    seed = 73121
cfg = MODEL_CFG()
model_cfg_dict[f'{cfg.model_name}_{suffix}'] = cfg