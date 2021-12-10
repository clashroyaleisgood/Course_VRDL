import numpy as np
# import pycocotools._mask as _mask
# reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py#L80
from pycocotools.mask import encode, area, toBbox
import os
import cv2
import json

# coco format: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
Mode = 'train'  # or 'valid'?
DatasetPath = r'HW3_Instance_Segmentation/dataset/' + Mode

SaveJsonPath = r'HW3_Instance_Segmentation/dataset/annotation.json'

images = []
annotations = []
categories = [{
    'id': 0,
    'name': 'null'
} , {
    'id': 1,
    'name': 'cuclei'
}]

ImageFolders = os.listdir(DatasetPath)  # list of(['TCGA-18-5592-01Z-00-DX1'])

<<<<<<< HEAD
for img_id, img_name in enumerate(ImageFolders):
=======
for img_id, img_name in enumerate(ImageFolders[:2]):
>>>>>>> f4bc22b6ba269fa17749e1551cbe86e5e202dc3e
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
        'file_name': os.path.join(Mode, img_name, 'images', img_name + '.png')
    }
    images += [image_data]
    # Process masks
    MaskImagesPath = os.listdir(MaskFolderPath)  # may have other data type
    for mask_id, mask_name in enumerate(MaskImagesPath):
        if not mask_name.endswith('.png'):
            continue
        MaskPath = os.path.join(MaskFolderPath, mask_name)
        mask = cv2.imread(MaskPath, cv2.IMREAD_GRAYSCALE)  # read as gray scale

        E = encode(np.asfortranarray(mask))
        A = area(E)
        bbox = toBbox(E)
        E['counts'] = E['counts'].decode('UTF-8')

        mask_data = {
            'id': mask_id,
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
    'annotations': annotations
}

with open(SaveJsonPath, 'w') as file:
    json.dump(data, file)
    # json.dump(data, file, indent=4)