import os
import hashlib
def get_md5_of_file(file_path):
    """
    计算文件的md5
    """
    md5 = None
    if os.path.isfile(file_path):
        f = open(file_path, 'rb')
        md5_obj = hashlib.md5()
        md5_obj.update(f.read())
        hash_code = md5_obj.hexdigest()
        f.close()
        md5 = str(hash_code).lower()
    return md5

file1 = "/opt/data/private/LYX/RS_detection/work_dirs/orcnn_van3_7_anchor_swa_3/checkpoints/swa_8-9.pkl"

md5_1 = get_md5_of_file(file1)

print(md5_1)