import os
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_path",
    default='../data/testa-3',
    type=str,
)
parser.add_argument(
    "--preprocessed_path",
    default='../data/testa_ms',
    type=str,
)
args = parser.parse_args()
import os
def alter(file,old_str,new_str):
 
    with open(file, "r", encoding="utf-8") as f1,open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)
dota_paht = '/'.join(args.preprocessed_path.split('/')[:-1])+'test_dota'
alter("configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py", '../data/testa-3', args.test_path)
alter("configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py", '../data/testa_dota', dota_paht)
alter("configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py", '../data/testa_ms', args.preprocessed_path)
alter("configs/orcnn_van3_for_test_1.py", '../data/testa_ms', args.preprocessed_path)
alter("configs/orcnn_van3_for_test_2.py", '../data/testa_ms', args.preprocessed_path)

if not os.path.exists(args.preprocessed_path):
    os.system('python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py')
else:
    if len(glob.glob(os.path.join(args.preprocessed_path,'test_1024_200_0.5-1.0-1.5/images/'))) == 0:
        os.system('python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py')

if not os.path.exists('submit_zips/orcnn_van3_for_test_1_epoch0.csv'):
    os.system('python tools/run_net.py --config-file configs/orcnn_van3_for_test_1.py --task test')
if not os.path.exists('submit_zips/orcnn_van3_for_test_2_epoch0.csv'):
    os.system('python tools/run_net.py --config-file configs/orcnn_van3_for_test_2.py --task test')
os.system('python merge.py')
