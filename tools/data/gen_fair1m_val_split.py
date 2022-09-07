"""script
python tools/data/gen_fair1m_val_split.py
"""

import os
import os.path as osp
import random
import numpy as np

val_path = "/yimian/fair1m/val/images"
out_dir = "/yimian/fair1m/splits"
os.makedirs(out_dir, exist_ok=True)

val_map = {
    'val10': 10,
    # 'val1k': 1000,
    # 'val2k': 2000,
    # 'val3k': 3000,
    # 'val4k': 4000,
    # 'val5k': 5000,
    # 'val6k': 6000,
    # 'val7k': 7000,
    # 'val8k': 8000,
    # 'val_full': np.inf
}

img_names = []
for img_root, dirs, files in os.walk(val_path):
    for f in files:
        img_names.append(f[:-4]+'\n')
random.shuffle(img_names)

for key, value in val_map.items():
    out_path = osp.join(out_dir, key+".txt")
    fo = open(out_path, "w+")
    img_num = min(value, len(img_names))
    fo.writelines(img_names[:img_num])
    fo.close()
