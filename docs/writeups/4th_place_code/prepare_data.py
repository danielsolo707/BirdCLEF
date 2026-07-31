import os
import multiprocessing as mp
from configs.config import data_cfg_dict

arg_parts = [
    'python utils/raw_data.py',
    'python utils/raw_semi_data.py'
]
pool = mp.Pool(processes=6)
pool.map(os.system, arg_parts)
pool.close()
pool.join()

arg_parts = []
for v in data_cfg_dict:
    if data_cfg_dict[v].USE_AUGMENT:
        arg_parts.append(f'python utils/aug_data.py --version {v}')
    else:
        arg_parts.append(f'python utils/data.py --version {v}')
pool = mp.Pool(processes=6)
pool.map(os.system, arg_parts)
pool.close()
pool.join()