import os
import os.path as osp
import xml.etree.ElementTree as ET
import pprint
pp = pprint.PrettyPrinter(depth=4)

data_root = "~/Downloads/DOTA-v2.0"
sub_dirs = ['train', 'val']

distributions = dict()
total_imgs = 0
for sub_dir in sub_dirs:
    sub_dir_path = osp.expanduser(osp.join(data_root, sub_dir))
    txt_files = [f for f in os.listdir(sub_dir_path)
                 if osp.isfile(osp.join(sub_dir_path, f))]
    for txt_file in txt_files:
        if not txt_file.endswith('.txt'):
            continue
        total_imgs += 1
        txt_path = osp.join(sub_dir_path, txt_file)
        f = open(txt_path, 'r')
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                if (len(splitlines) < 9):
                    continue
                if (len(splitlines) >= 9):
                    cls_name = splitlines[8]
                    if cls_name not in distributions:
                        distributions[cls_name] = 1
                    else:
                        distributions[cls_name] += 1
            else:
                break

pp.pprint(distributions)
total_instances = sum(distributions.values())
print('\ntotal_imgs:', total_imgs)
print('\ntotal_instances:', total_instances)

jdet_coarse2fine_mappings = {
    "Airplane": ["plane"
        ],
    "Ship": ["ship"
        ],
    "Vehicle": ["large-vehicle", "small-vehicle"
        ],
    "Basketball Court": ["basketball-court"],
    "Tennis Court": ["tennis-court"],
    "Football Field": ["soccer-ball-field"],
    "Baseball Field": ["baseball-diamond"],
    # "Intersection": [""],
    "Roundabout": ["roundabout"],
    "Bridge": ["bridge"],
    "Others": ["airport", "container-crane", "ground-track-field", "harbor",
               "helicopter", "helipad", "storage-tank", "swimming-pool"],
}

dota2jdet_distributions = {}
for key, value in jdet_coarse2fine_mappings.items():
    if key not in dota2jdet_distributions:
        dota2jdet_distributions[key] = 0
    for v in value:
        if v in distributions:
            dota2jdet_distributions[key] += distributions[v]

pp.pprint(dota2jdet_distributions)
total_dota2jdet_instances = sum(dota2jdet_distributions.values())
print('\ntotal_dota2jdet_instances:', total_dota2jdet_instances)