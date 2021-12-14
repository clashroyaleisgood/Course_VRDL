import numpy as np
# import pycocotools._mask as _mask
# reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py#L80
from pycocotools.mask import encode, area, toBbox
import os
import cv2
import json
from tqdm import tqdm

# coco format: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

DatasetPath = r'HW3_Instance_Segmentation/dataset/train'

SaveJsonPath = r'HW3_Instance_Segmentation/dataset/annot/train_annotation.json'

if not os.path.isdir(r'HW3_Instance_Segmentation/dataset/annot'):
    os.mkdir(r'HW3_Instance_Segmentation/dataset/annot')
# -----------------------------------------------------------------------------

images = []
annotations = []
categories = [{
    'id': 1,
    'name': 'nuclei'
}]

ImageFolders = os.listdir(DatasetPath)  # list of(['TCGA-18-5592-01Z-00-DX1'])
running_mask_id = 0

for img_id, img_name in enumerate(tqdm(ImageFolders)):
    img_id += 1
    ImageFolder = os.path.join(DatasetPath, img_name)  # path to TCGA-18-5592-01Z-00-DX1/
    ImagePath = os.path.join(ImageFolder, 'images', img_name + '.png')
    MaskFolderPath = os.path.join(ImageFolder, 'masks')

    # Process image
    image = cv2.imread(ImagePath)
    height, width = image.shape[:2]

    image_data = {
        'id': img_id,
        'width': width,
        'height': height,
        'file_name': os.path.join(img_name, 'images', img_name + '.png')
    }
    images += [image_data]
    # Process masks
    MaskImagesPath = os.listdir(MaskFolderPath)  # may have other data type
    for _, mask_name in enumerate(MaskImagesPath):
        if not mask_name.endswith('.png'):
            continue
        MaskPath = os.path.join(MaskFolderPath, mask_name)
        mask = cv2.imread(MaskPath, cv2.IMREAD_GRAYSCALE)  # read as gray scale

        E = encode(np.asfortranarray(mask))
        A = area(E)
        bbox = toBbox(E)
        E['counts'] = E['counts'].decode('UTF-8')

        running_mask_id += 1
        mask_data = {
            'id': running_mask_id,
            'category_id': 1,
            'is_crowd': 0,
            'image_id': img_id,
            'segmentation': E,
            'area': float(A),
            'bbox': [*bbox]
        }
        annotations += [mask_data]

data = {
    'images': images,
    'annotations': annotations,
    'categories': categories
}

with open(SaveJsonPath, 'w') as file:
    json.dump(data, file)
    # json.dump(data, file, indent=4)
