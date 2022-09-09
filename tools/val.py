import glob
import os
import numpy as np
import argparse
from jdet.data.devkits.voc_eval import voc_eval_dota
from jdet.ops.nms_poly import iou_poly
from tqdm import tqdm
from lxml import etree
from jdet.models.boxes.box_ops import poly_to_rotated_box_single, rotated_box_to_poly_single

FAIR1M_1_5_CLASSES = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 
        "Football_Field", "Baseball_Field", 'Intersection', 'Roundabout', 'Bridge']

def read_csv_to_numpy(submit_csvfile_path:str):
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


def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag not in ['object', 'point'] :
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}
    

def xml_to_dict(xml_path:str):
    with open(xml_path,"rb") as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    obj = parse_xml_to_dict(xml)
    return obj


def read_xml(xml_path:str):
    xml_dict = xml_to_dict(xml_path)
    file_name = xml_dict["annotation"]["source"]["filename"]
    image_idx = int(file_name.split(".")[0])
    labels = []
    point_list = []
    for object in xml_dict["annotation"]["objects"]["object"]:
        point_list.append(object["points"])
        label_class_name = object["possibleresult"]["name"]
        labels.append(FAIR1M_1_5_CLASSES.index(label_class_name.replace(" ", "_")) + 1)
    point_list = [x["point"][0:4] for x in point_list]
    for i in range(len(point_list)):
        temp = []
        for j in range(4):
            p = [float(x) for x in point_list[i][j].split(",")]
            temp.extend(p)
        point_list[i] = temp
    return image_idx, labels, point_list

def read_xml_to_numpy(xml_list):
    gts = []
    diffcult_polys = {}
    for xml_path in xml_list:
        image_idx, det_labels, det_point_list = read_xml(xml_path)
        # for i, point in enumerate(det_point_list):
        #     det_point_list[i] = poly_to_rotated_box_single(point)
        #     det_point_list[i] = rotated_box_to_poly_single(det_point_list[i])
        det_labels = np.array(det_labels)
        det_point_numpy = np.array(det_point_list)
        idx1 = np.ones((len(det_labels), 1)) * image_idx
        det = np.concatenate([idx1, det_point_list, det_labels.reshape(-1,1)], axis=1)
        gts.append(det)
    return np.concatenate(gts)


def evaluate(submit_csvfile_path, read_from_xml = False):
    xml_list = glob.glob("val/*.xml")
    print("Calculating mAP......")
    print(submit_csvfile_path)
    # dets[0].shape = 11 : idx, ploys, score, label
    # gts[0].shape = 10  : idx, ploys, label
    dets = read_csv_to_numpy(submit_csvfile_path)
    if read_from_xml:
        gts = read_xml_to_numpy(xml_list)
    else:
        gts_file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "gts.npy") 
        assert os.path.exists(gts_file_path),  "file {} not exits".format(gts_file_path)
        gts = np.load(gts_file_path)
    if len(dets) == 0:
        aps = {}
        for i, classname in tqdm(enumerate(FAIR1M_1_5_CLASSES), total = len(FAIR1M_1_5_CLASSES)):
            aps[classname + "_AP"] = 0 
        map = sum(list(aps.values())) / len(aps)
        aps["meanAP"]=map
        return aps
    
    aps = {}
    for i, classname in tqdm(enumerate(FAIR1M_1_5_CLASSES), total = len(FAIR1M_1_5_CLASSES)):
        # get the all objects for the classes
        c_dets = dets[dets[:,-1]==(i+1)][:,:-1]
        c_gts = gts[gts[:,-1]==(i+1)][:,:-1]

        img_idx = gts[:,0].copy()
        classname_gts = {}
        for idx in np.unique(img_idx):
            # g is the same object for a image
            g = c_gts[c_gts[:,0]==idx,:][:,1:]
            dg = np.array([], dtype = np.float64).reshape(-1,8)
            diffculty = np.zeros(g.shape[0]+dg.shape[0])
            diffculty[int(g.shape[0]):]=1
            diffculty = diffculty.astype(bool)
            g = np.concatenate([g,dg])
            classname_gts[idx] = {"box":g.copy(),"det":[False for i in range(len(g))],'difficult':diffculty.copy()}
        rec, prec, ap = voc_eval_dota(c_dets,classname_gts,iou_func=iou_poly)
        aps[classname + "_AP"]=ap 
    map = sum(list(aps.values()))/len(aps)
    aps["meanAP"]=map
    return aps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvfile_path",
        default="/opt/data/private/val_func/9120.csv",
        type=str,
    )
    args = parser.parse_args()
    assert os.path.exists(args.csvfile_path), "file {} not exits.".format(args.csvfile_path)
    result = evaluate(args.csvfile_path)
    print("\nValidation Score:", result["meanAP"])
    result.pop("meanAP")
    for k, v in result.items():
        print("{:.5f}".format(v), "({})".format(k))

if __name__ == "__main__":
    main()
