import os
from configs.config import semi_model_cfg_dict, model_cfg_dict

for v in semi_model_cfg_dict:
    os.system(f'python utils/model.py --version {v} --mode pretrain')
os.system('python utils/semi_infer.py')

for v in model_cfg_dict:
    os.system(f'python utils/model.py --version {v} --mode train')