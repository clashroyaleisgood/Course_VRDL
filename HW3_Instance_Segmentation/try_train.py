import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

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
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.INPUT.MASK_FORMAT = 'bitmask'  # added: RLE form
cfg.DATASETS.TRAIN = ("Nuclei_train",)
cfg.DATASETS.TEST = ("Nuclei_test",)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.SOLVER.STEPS = [900, 1000]
cfg.SOLVER.MAX_ITER = 1200    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset

# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.SOLVER.CHECKPOINT_PERIOD = 40  # 40 epoch 存一次 weight
cfg.OUTPUT_DIR = r'HW3_Instance_Segmentation/exp'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

trainer.train()
