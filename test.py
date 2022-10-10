import os
if not os.path.exists('submit_zips/orcnn_van3_for_test_1_epoch0.csv'):
    os.system('python tools/run_net.py --config-file configs/orcnn_van3_for_test_1.py --task test')
if not os.path.exists('submit_zips/orcnn_van3_for_test_2_epoch0.csv'):
    os.system('python tools/run_net.py --config-file configs/orcnn_van3_for_test_2.py --task test')
os.system('python merge.py')