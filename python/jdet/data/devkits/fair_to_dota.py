import os
import cv2
from xml.dom.minidom import parse
from tqdm import tqdm
import sys
import random

def solve_xml(src, tar):
    domTree = parse(src)
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object")
    box_list=[]
    for obj in objects:
        name=obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data
        points=obj.getElementsByTagName("points")[0].getElementsByTagName("point")
        bbox=[]
        for point in points[:4]:
            x=point.childNodes[0].data.split(",")[0]
            y=point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        box_list.append({"name":name, "bbox":bbox})

    file=open(tar,'w')
    print("imagesource:GoogleEarth",file=file)
    print("gsd:0.0",file=file)
    for box in box_list:
        ss=""
        for f in box["bbox"]:
            ss+=str(f)+" "
        name=  box["name"]
        name = name.replace(" ", "_")
        ss+=name+" 0"
        print(ss,file=file)
    file.close()

def fair_to_dota(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)

    tasks = []
    for root, dirs, files in os.walk(os.path.join(in_path, "images")):
        for f in files:
            src=os.path.join(root, f)
            tar="P"+f[:-4].zfill(4)+".png"
            tar=os.path.join(out_path,"images", tar)
            tasks.append((src, tar))
    print("processing images")
    for task in tqdm(tasks):
        file = cv2.imread(task[0], 1)
        cv2.imwrite(task[1], file)

    if (os.path.exists(os.path.join(in_path, "labelXml"))):
        os.makedirs(os.path.join(out_path, "labelTxt"), exist_ok=True)
        tasks = []
        for root, dirs, files in os.walk(os.path.join(in_path, "labelXml")):
            for f in files:
                src=os.path.join(root, f)
                tar="P"+f[:-4].zfill(4)+".txt"
                tar=os.path.join(out_path,"labelTxt", tar)
                tasks.append((src, tar))
        print("processing labels")
        for task in tqdm(tasks):
            solve_xml(task[0], task[1])

def fair_to_dota_select(in_path, out_path, split_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)

    # opening the file in read mode
    split_file = open(split_path, "r")
    # reading the file
    file_names = split_file.read().split("\n")
    file_names.remove("")
    # print("file_names:", file_names)
    split_file.close()

    # # extract file names
    # img_names = set()
    # for img_root, dirs, files in os.walk(os.path.join(in_path, "images")):
    #     for f in files:
    #         img_names.add(f[:-4])
    # xml_names = set()

    # for xml_root, dirs, files in os.walk(os.path.join(in_path, "labelXml")):
    #     for f in files:
    #         xml_names.add(f[:-4])
    # if img_names == xml_names:
    #     file_names = list(img_names)
    # else:
    #     raise ValueError("imgs and xmls are not matched !!!")

    # # select part of files to process
    # if (select_num is not None) and (select_num < len(file_names)):
    #     random.shuffle(file_names)
    #     file_names = file_names[:select_num]


    img_root = os.path.join(in_path, "images")
    img_tasks = []
    for f in file_names:
        img_src = os.path.join(img_root, f+".tif")
        img_tar = "P"+f.zfill(4)+".png"
        img_tar = os.path.join(out_path, "images", img_tar)
        img_tasks.append((img_src, img_tar))

    print("processing images")
    for task in tqdm(img_tasks):
        file = cv2.imread(task[0], 1)
        cv2.imwrite(task[1], file)

    # for root, dirs, files in os.walk(os.path.join(in_path, "images")):
    #     for f in files:
    #         src=os.path.join(root, f)
    #         tar="P"+f[:-4].zfill(4)+".png"
    #         tar=os.path.join(out_path,"images", tar)
    #         tasks.append((src, tar))

    # print("processing images")
    # for task in tqdm(tasks):
    #     file = cv2.imread(task[0], 1)
    #     cv2.imwrite(task[1], file)

    xml_root = os.path.join(in_path, "labelXml")
    if (os.path.exists(os.path.join(in_path, "labelXml"))):
        os.makedirs(os.path.join(out_path, "labelTxt"), exist_ok=True)
        xml_tasks = []
        for f in file_names:
            xml_src = os.path.join(xml_root, f+".xml")
            xml_tar = "P"+f[:-4].zfill(4)+".txt"
            xml_tar=os.path.join(out_path, "labelTxt", xml_tar)
            xml_tasks.append((xml_src, xml_tar))
        print("processing labels")
        for task in tqdm(xml_tasks):
            solve_xml(task[0], task[1])

    # if (os.path.exists(os.path.join(in_path, "labelXml"))):
    #     os.makedirs(os.path.join(out_path, "labelTxt"), exist_ok=True)
    #     tasks = []
    #     for root, dirs, files in os.walk(os.path.join(in_path, "labelXml")):
    #         for f in files:
    #             src=os.path.join(root, f)
    #             tar="P"+f[:-4].zfill(4)+".txt"
    #             tar=os.path.join(out_path,"labelTxt", tar)
    #             tasks.append((src, tar))
    #     print("processing labels")
    #     for task in tqdm(tasks):
    #         solve_xml(task[0], task[1])

if __name__ == '__main__':
    src = sys.argv[1]
    tar = sys.argv[2]
    fair_to_dota(src, tar)
