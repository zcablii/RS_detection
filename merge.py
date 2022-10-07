import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
import argparse
import glob
from typing import Union
import pandas as pd
sys.path.append("/opt/data/private/github_push/RS_detection/tools")
# from val import test_evaluate

FAIR1M_1_5_CLASSES = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 
        "Football_Field", "Baseball_Field", 'Intersection', 'Roundabout', 'Bridge']

def nms(boxes, thresh):
    areas = np.prod(boxes[:,2:4] - boxes[:,:2], axis=1)
    order = boxes[:,4].argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tl = np.maximum(boxes[i][:2], boxes[order[1:]][:,:2])
        br = np.minimum(boxes[i][2:4], boxes[order[1:]][:,2:4])
        overlaps = np.prod(br - tl, axis=1) * (br > tl).all(axis=1)
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)
        inds = np.where(ious < thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def soft_nms(boxes, method = 'linear', thresh = 0.3, Nt = 0.6, sigma = 0.5):
    areas = np.prod(boxes[:,2:4] - boxes[:,:2], axis=1)
    order = boxes[:,4].argsort()[::-1]
    score = boxes[:,4]
    keep = []
    while order.size>0:
        i = order[0]
        keep.append(i)
        tl = np.maximum(boxes[i][:2], boxes[order[1:]][:,:2])
        br = np.minimum(boxes[i][2:4], boxes[order[1:]][:,2:4])
        overlaps = np.prod(br - tl, axis = 1) * (br > tl).all(axis = 1)
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)
        weight = np.ones_like(overlaps)
        if method == 'linear':
            weight[np.where(ious > Nt)] = 1 - ious[np.where(ious > Nt)]
        elif method == 'gaussian':
            weight = np.exp( -(ious * ious) / sigma)
        else:
            weight[np.where(ious > Nt)] = 0
        score[order[1:]] = score[order[1:]] * weight
        inds = np.where(score[order[1:]] > thresh)[0]
        order = order[inds + 1]
    return np.array(keep)
    
def read_csv_to_numpy(submit_csvfile_path:str):
    """
    读取csv文件 转换成numpy格式 (image_id, poly_points, score, classify)
    """
    detection_numpy = []
    with open(submit_csvfile_path, "r") as f:
        for each_line in f.readlines():
            each_line_split = each_line.strip().split(",")
            assert len(each_line_split) == 11, "csv file format error"
            image_idx = int(each_line_split[0].split(".")[0])
            label_class_name = each_line_split[1]
            assert label_class_name in FAIR1M_1_5_CLASSES, "laebl name not matched"
            label_class_idx = FAIR1M_1_5_CLASSES.index(label_class_name) + 1
            polys = [float(x) for x in each_line_split[2:-1]]
            detection_score = float(each_line_split[-1])
            detection_numpy.append([image_idx, *polys, detection_score, label_class_idx])
    return np.array(detection_numpy)


def poly2obb(polys):
    """
    :param polys:array[
        [x1, y1, x2, y2, x3, y3, x4, y4]....
    ]
    """
    order = polys.shape[:-1]
    num_points = polys.shape[-1] // 2
    polys = polys.reshape(-1, num_points, 2)
    polys = polys.astype(np.float32)

    obboxes = []
    for poly in polys:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * np.pi
        obboxes.append([x, y, w, h, theta])
    if not obboxes:
        obboxes = np.zeros((0, 5))
    else:
        obboxes = np.array(obboxes)
    obboxes = obboxes.reshape(*order, 5)
    return np.array(obboxes)


def obb2hbb(obboxes):
    """
    :param obboxes:[x, y, w, h, theta]
    """
    center, w, h, theta, _ = np.split(obboxes, [2, 3, 4, 5], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = np.abs(w/2 * Cos) + np.abs(h/2 * Sin)
    y_bias = np.abs(w/2 * Sin) + np.abs(h/2 * Cos)
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return np.concatenate([center-bias, center+bias], axis=-1)


def save_to_csv(data, output_path:str):
    with open(output_path, "w") as f:
        for each in data:
            temp = []
            temp.append(f"{int(each[0])}.tif")
            temp.append(FAIR1M_1_5_CLASSES[int(each[10]) - 1])
            for i in each[1:1+8]:
                temp.append("{:.4f}".format(i))
            temp.append("{:.4f}".format(each[9]))
            f.write(",".join(temp))
            f.write("\n")


def merge_csv_with_class(data_list:list, thresh:Union[int, dict], soft_param=[0.3,0.6]):
    """
    对两个csv文件进行融合 每张图每种类别proposal单独进行nms
    """
    is_thresh_dic = (type(thresh) == type({}))

    image_id_list = np.unique(data_list[0][:,0])
    
    # for each image
    result = []
    print("merging result")
    for image_id in tqdm(image_id_list):
        image_dets = []
        for data in data_list:
            image_dets.append(data[data[:,0] == image_id,:])
        image_dets = np.concatenate(image_dets)
        for class_idx in range(10):
            # for each class
            class_name = FAIR1M_1_5_CLASSES[class_idx]
            class_thresh = thresh[class_name] if is_thresh_dic else thresh
            
            image_class_dets = image_dets[image_dets[:,-1] == class_idx + 1]
            hbb_data = obb2hbb(poly2obb(image_class_dets[:,1:9]))
            proposal = np.concatenate([hbb_data, image_class_dets[:,9:10]], axis=1)
            # valid_idx = soft_nms(boxes = proposal, thresh = soft_param[0], Nt=soft_param[1])
            valid_idx = nms(boxes = proposal, thresh = class_thresh)
            if valid_idx.shape[0] > 0:
                result.append(image_class_dets[valid_idx,:])
    result = np.concatenate(result)
    return result


def merge_csv_without_class(data_list:list, thresh:int):
    """
    对两个csv文件进行融合 每张图所有proposal进行nms
    """
    image_id_list = np.unique(data_list[0][:,0])
    result = []
    print("merging result")
    for image_id in tqdm(image_id_list):
        image_dets = []
        for data in data_list:
            image_dets.append(data[data[:,0] == image_id,:])
        image_dets = np.concatenate(image_dets)
        hbb_data = obb2hbb(poly2obb(image_dets[:,1:9]))
        proposal = np.concatenate([hbb_data, image_dets[:,9:10]], axis=1)
        valid_idx = nms(proposal, thresh)
        if valid_idx.shape[0] > 0:
            result.append(image_dets[valid_idx,:])
    result = np.concatenate(result)
    return result


def judge_exist(path_list):
    # check the path exist
    assert len(path_list) > 0, "need 2 path at least"
    for each_path in path_list:
        assert os.path.exists(each_path),  "file {} not exits".format(each_path)

# def meshgrid_find_param(thresh_dict, data_list):
#     """
#     根据测试数据对参数进行调节(nms)
#     """
#     for k, _ in thresh_dict.items():
#         best_thresh = 0
#         max_value = 0
#         for thresh in np.linspace(0,1,21):
#             thresh_dict[k] = thresh
#             data = merge_csv_with_class(data_list, thresh_dict)
#             result = test_evaluate(data)
#             if result[k + "_AP"] > max_value:
#                 max_value = result[k + "_AP"]
#                 best_thresh = thresh
#         thresh_dict[k] = best_thresh
#     return thresh_dict

# def soft_nms_find_param(data_list):
#     best_param = [0, 0]
#     max_map = 0
#     for thresh in np.linspace(0,1,11):
#         for Nt in np.linspace(0,1,11):
#             data = merge_csv_with_class(data_list, 0, [thresh, Nt])
#             result = test_evaluate(data)
#             print(max_map, result["meanAP"])
#             if result["meanAP"] > max_map:
#                 max_map = result["meanAP"]
#                 best_param = [thresh, Nt]
#     print(f"find best param thresh = {best_param[0]}, Nt = {best_param[1]}, max_map = {max_map}")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--is_compare",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--is_output",
        default=True,
        type=bool
    )
    args = parser.parse_args()

    # 载入预测结果
    merge_path_list = glob.glob("./csv_merge/*.csv")
    # merge_path_list = glob.glob("/opt/data/private/model_merge/results/*.csv")

    # 检测地址是否出错
    judge_exist(merge_path_list)

    # 读取所有的文件
    data_list = [read_csv_to_numpy(path) for path in merge_path_list]

    # 通过nms进行融合
    thresh_dict = {
              'Airplane': 0.7,
                  'Ship': 0.7,
               'Vehicle': 0.7,
      'Basketball_Court': 0.3,
          'Tennis_Court': 0.2, 
        "Football_Field": 0.2,
        "Baseball_Field": 0.2,
          'Intersection': 0.4,
            'Roundabout': 0.2,
                'Bridge': 0.5
    }
    # thresh_dict = {
    #           'Airplane': 0.75,
    #               'Ship': 0.70,
    #            'Vehicle': 0.70,
    #   'Basketball_Court': 0.25,
    #       'Tennis_Court': 0.15, 
    #     "Football_Field": 0.20,
    #     "Baseball_Field": 0.20,
    #       'Intersection': 0.35,
    #         'Roundabout': 0.00,
    #             'Bridge': 0.50
    # }
    # 网络搜索最好的阈值 for nms
    # thresh_dict = meshgrid_find_param(thresh_dict, data_list)

    # 网络搜索最好的参数 for soft-nms
    # soft_nms_find_param(data_list)

    # 这里可以传入统一的thresh
    result = merge_csv_with_class(data_list, thresh_dict)
    
    # 如果不按类别进行nms
    # result = merge_csv_without_class(data_list, 0.9)
    save_to_csv(result, "./csv_merge/merged_result.csv")
    
    return
        
if __name__ == "__main__":
    main()
