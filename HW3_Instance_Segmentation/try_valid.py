import json
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

For_Vis = False  # maks annot for vis.py
For_Vis = True
SaveVisJsonPath = r'HW3_Instance_Segmentation/predict_Vis_annotation.json'
SaveJsonPath = r'HW3_Instance_Segmentation/predict_annotation.json'

from detectron2.data.datasets import register_coco_instances
register_coco_instances("Nuclei_train", {},
                        json_file=r'HW3_Instance_Segmentation/dataset/annot/train_annotation.json',
                        image_root=r'HW3_Instance_Segmentation/dataset/train')
register_coco_instances("Nuclei_test", {},
                        json_file=r'HW3_Instance_Segmentation/dataset/annot/test_annotation.json',
                        image_root=r'HW3_Instance_Segmentation/dataset/test')

# 設定類別
from detectron2.data import MetadataCatalog
MetadataCatalog.get("Nuclei_train").thing_classes = ['nuclei']
MetadataCatalog.get("Nuclei_test").thing_classes = ['nuclei']

import os
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.INPUT.MASK_FORMAT = 'bitmask'  # added: RLE form
# cfg.DATASETS.TRAIN = ("Nuclei_train",)
cfg.DATASETS.TEST = ("Nuclei_test",)
cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 1200    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.SOLVER.CHECKPOINT_PERIOD = 40  # 40 epoch 存一次 weight
cfg.OUTPUT_DIR = r'HW3_Instance_Segmentation/exp'

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode, Visualizer
dataset_dicts = DatasetCatalog.get("Nuclei_test")

# ----------------------------------------------------------------------------
coco_form_mask = []
running_mask_id = 0

import numpy as np
from pycocotools.mask import encode, area, toBbox
for d in tqdm(dataset_dicts):
    im = cv2.imread(d["file_name"])
    filename = os.path.basename(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # v = Visualizer(im[:, :, ::-1],
    #                metadata=MetadataCatalog.get('Nuclei_train'),
    #                scale=0.5,
    #                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    # )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite(f'output_infer/{filename}', out.get_image()[:, :, ::-1])

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
            mask['id'] =  running_mask_id

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

    # rects = outputs["instances"][outputs["instances"].pred_classes == 0].pred_boxes.tensor.cpu().numpy()
    # scores = outputs["instances"][outputs["instances"].pred_classes == 0].scores.tolist()
    # classes = outputs["instances"][outputs["instances"].pred_classes == 0].pred_classes.tolist()

    