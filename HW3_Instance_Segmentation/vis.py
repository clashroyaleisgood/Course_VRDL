import enum
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

SaveVisFolder = r'HW3_Instance_Segmentation/VisJson'
if not os.path.isdir(SaveVisFolder):
    os.mkdir(SaveVisFolder)

# register_coco_instances("Nuclei_train", {},
#     json_file = r'HW3_Instance_Segmentation/dataset/annot/train_annotation.json',
#     image_root =   r'HW3_Instance_Segmentation/dataset/train')
register_coco_instances("Nuclei_test", {},
    json_file = r'HW3_Instance_Segmentation/predict_Vis_annotation.json',
    image_root =   r'HW3_Instance_Segmentation/dataset/test')

# 設定類別
from detectron2.data import MetadataCatalog
# MetadataCatalog.get("Nuclei_train").thing_classes = ['nuclei']
MetadataCatalog.get("Nuclei_test").thing_classes = ['nuclei']

# ------------------------------------------------------------

dataset_dicts = DatasetCatalog.get("Nuclei_test")
# print(dataset_dicts.keys())

import cv2
import os
import random
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer

for d in tqdm(dataset_dicts):
    # print(d)
    img = cv2.imread(d["file_name"])
    filename = os.path.basename(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('Nuclei_test'), scale=1)
    vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow('image', vis.get_image()[:, :, ::-1])
    cv2.imwrite(os.path.join(SaveVisFolder, filename), vis.get_image()[:, :, ::-1])
