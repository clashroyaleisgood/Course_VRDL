import numpy as np
# import pycocotools._mask as _mask
# reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py#L80
from pycocotools.mask import encode, area, toBbox
import os
import cv2
import json

# coco format: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

DatasetPath = r'HW3_Instance_Segmentation/dataset/test'
TestImageIDPath = r'HW3_Instance_Segmentation/dataset/test_img_ids.json'

SaveJsonPath = r'HW3_Instance_Segmentation/dataset/annot/test_annotation.json'

if not os.path.isdir(r'HW3_Instance_Segmentation/dataset/annot'):
    os.mkdir(r'HW3_Instance_Segmentation/dataset/annot')
# -----------------------------------------------------------------------------

images = []
categories = [{
    'id': 1,
    'name': 'nuclei'
}]

# ImageFolders = os.listdir(DatasetPath)  # list of(['TCGA-18-5592-01Z-00-DX1'])

# for img_id, img_name in enumerate(ImageFolders):
#     if not img_name.endswith('.png'):
#         continue
#     ImagePath = os.path.join(DatasetPath, img_name)  # path to TCGA-18-5592-01Z-00-DX1.png
#     # ImagePath = os.path.join(ImageFolder, 'images', img_name + '.png')
#     # MaskFolderPath = os.path.join(ImageFolder, 'masks')

#     # Process image
#     image = cv2.imread(ImagePath)
#     height, width = image.shape[:2]

#     image_data = {
#         'id': img_id,
#         'width': width,
#         'height': height,
#         'file_name': img_name
#     }
#     images += [image_data]

with open(TestImageIDPath, 'r') as file:
    images = json.load(file)

data = {
    'images': images,
    'categories': categories
}

with open(SaveJsonPath, 'w') as file:
    json.dump(data, file)
    # json.dump(data, file, indent=4)
