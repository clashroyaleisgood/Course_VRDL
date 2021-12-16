import json
import cv2
import os
from tqdm import tqdm
import numpy as np
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from pycocotools.mask import encode, area, toBbox

For_Vis = True  # maks annot for vis.py
SaveVisJsonPath = r'HW3_Instance_Segmentation/predict_Vis_annotation.json'
SaveJsonPath = r'HW3_Instance_Segmentation/predict_annotation.json'

register_coco_instances("Nuclei_train", {},
                        json_file=r'HW3_Instance_Segmentation/dataset/annot/train_annotation.json',
                        image_root=r'HW3_Instance_Segmentation/dataset/train')
register_coco_instances("Nuclei_test", {},
                        json_file=r'HW3_Instance_Segmentation/dataset/annot/test_annotation.json',
                        image_root=r'HW3_Instance_Segmentation/dataset/test')

# 設定類別
MetadataCatalog.get("Nuclei_train").thing_classes = ['nuclei']
MetadataCatalog.get("Nuclei_test").thing_classes = ['nuclei']

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

cfg.INPUT.MASK_FORMAT = 'bitmask'  # added: RLE form
cfg.INPUT.MIN_SIZE_TEST = 1000
# cfg.DATASETS.TRAIN = ("Nuclei_train",)
cfg.DATASETS.TEST = ("Nuclei_test",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.SOLVER.CHECKPOINT_PERIOD = 40  # 40 epoch 存一次 weight
cfg.OUTPUT_DIR = r'HW3_Instance_Segmentation/exp'

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0002099.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
predictor = DefaultPredictor(cfg)

dataset_dicts = DatasetCatalog.get("Nuclei_test")

# ----------------------------------------------------------------------------
coco_form_mask = []
running_mask_id = 0

for d in tqdm(dataset_dicts):
    im = cv2.imread(d["file_name"])
    filename = os.path.basename(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    masks = outputs["instances"].to("cpu")
    bbox = masks.pred_boxes.tensor.numpy()  # (N, 4): [left, top, width, height]
    conf = masks.scores.tolist()
    pred_mask = masks.pred_masks.numpy().astype(np.uint8)
    masks_num = len(conf)

    bbox[:, 2:4] -= bbox[:, 0:2]  # [left, top, right, down] to [left, top, width, height]

    for i in range(masks_num):
        E = encode(np.asfortranarray(pred_mask[i]))
        E['counts'] = E['counts'].decode('UTF-8')
        # bbox = [e.item() for e in bbox[i]]

        mask = {
            'image_id': d['image_id'],
            'bbox': [e.item() for e in bbox[i]],
            'score': conf[i],
            'category_id': 1,
            'segmentation': E,
        }
        # print(mask)
        if For_Vis:
            running_mask_id += 1
            mask['id'] = running_mask_id

        coco_form_mask += [mask]

if For_Vis:
    data = None
    with open(MetadataCatalog.get("Nuclei_test").json_file) as file:
        data = json.load(file)
    # data = json.load(MetadataCatalog.get("Nuclei_test").json_file)
    data['annotations'] = coco_form_mask
    with open(SaveVisJsonPath, 'w') as file:
        json.dump(data, file)

with open(SaveJsonPath, 'w') as file:
    json.dump(coco_form_mask, file)
