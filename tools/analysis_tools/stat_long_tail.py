"""script
python tools/analysis_tools/stat_long_tail.py
"""

from collections import OrderedDict

fair1m_1_5_dict = OrderedDict({
    'Airplane': 44248,
    'Ship': 58982,
    'Vehicle': 469857,
    'Basketball_Court': 1707,
    'Tennis_Court': 4337,
    'Football_Field': 1357,
    'Baseball_Field': 1519,
    'Intersection': 10199,
    'Roundabout': 841,
    'Bridge': 1730,
})

total_num_fair1m = 0
for key, value in fair1m_1_5_dict.items():
    total_num_fair1m += fair1m_1_5_dict[key]
print("total_num_fair1m:", total_num_fair1m)

for key, value in fair1m_1_5_dict.items():
    print(key + " vs Others:", "1 :", (total_num_fair1m - value) / value)

from jdet.data.lvis import LVIS_CATEGORIES

total_num_lvis = 0
for cls_dict in LVIS_CATEGORIES:
    total_num_lvis += cls_dict['instance_count']
print("total_num_lvis:", total_num_lvis)

# for key, value in fair1m_1_5_dict.items():
#     print(key + " vs Others:", "1 :", (total_num_fair1m - value) / value)
lvis_ratios = []
for cls_dict in LVIS_CATEGORIES:
    instance_count = cls_dict['instance_count']
    ratio = (total_num_lvis - instance_count) / instance_count
    lvis_ratios.append(ratio)
lvis_ratios.sort()
beg_idx = int(0.1 * len(lvis_ratios))
print(lvis_ratios[beg_idx:beg_idx+10])
