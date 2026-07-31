import multiprocessing as mp
import os
import pandas as pd
import time
from configs.config import settings

parts = [0, 1]

arg_parts = [f'python utils/test.py --part {p} --n_parts {len(parts)}' for p in parts]
processes = []
for arg_part in arg_parts:
    p = mp.Process(target=os.system, args=(arg_part, ))
    p.start()
    processes.append(p)

sub = []
for p in parts:
    while not os.path.exists(f"{settings['data_cache_dir']}/finished{p}"):
        time.sleep(0.5)
    sub.append(pd.read_csv(f"{settings['data_cache_dir']}/submission_part{p}.csv"))

if not os.path.exists(settings['submission_dir']):
    os.mkdir(settings['submission_dir'])
sub = pd.concat(sub).reset_index(drop=True)
sub.to_csv(f"{settings['submission_dir']}/submission.csv", index=False)

for part in parts:
    processes[part].terminate()
    os.system(f"rm {settings['data_cache_dir']}/finished{part}")