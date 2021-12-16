# Homework3 - Instance Segmentation

You can see report [here](report/report.md) in markdown format, or [pdf](report/VRDL_HW3_309553018_Report.pdf) format which is exported by [Typora](https://typora.io/).

Competetion: https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5

Result score on testing dataset: **0.24303**

with help of [Detectron2](https://github.com/facebookresearch/detectron2)

Model weight: [model_final.pth]()

## Table of Contents

- [Homework3 - Instance Segmentation](#homework3---instance-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Environment](#environment)
  - [Code](#code)
  - [Dataset](#dataset)
  - [Preprocessing](#preprocessing)
  - [Model](#model)
  - [Architecture](#architecture)
  - [Train or Test](#train-or-test)
    - [Testing (inference)](#testing-inference)
    - [Training](#training)
  - [Reference (helps)](#reference-helps)

## Environment
Install CUDA, cuDNN

Install Pytorch/ Detectron2:
```
conda create -n Det python=3.7
conda activate Det
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
```

Install modules:
```python=
pip install tqdm numpy opencv-python
```

or simply
```
pip install -r Course_VRDL/HW3_Instance_Segmentation/requirements.txt
```

## Code
Download code with the following command:  
`$ git clone https://github.com/clashroyaleisgood/Course_VRDL.git`

## Dataset
Get data from this [competition](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5), [Dataset Link](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view?usp=sharing)

Put Training data to `HW3_Instance_Segmentation/dataset/`
```
Course_VRDL/
    └── dataset/
        ├── train/
        │   ├── TCGA-18-5592-01Z-00-DX1/
        │   └── TCGA-21-5784-01Z-00-DX1/...
        ├── test/
        └── test_img_ids.json
```

## Preprocessing
> mask sure that you are in conda env: **Det**  
> with `conda activate Det`

```
# write training annotation with COCO format(.json)
~/Course_VRDL$ python HW3_Instance_Segmentation/mask2coco.py
# write testing annotation with COCO format(.json)
~/Course_VRDL$ python HW3_Instance_Segmentation/testdata_coco.py
```
reference: [COCO format](https://cocodataset.org/#format-data)

## Model
Get weight with this [link](https://drive.google.com/file/d/18EZakC0NJaug1ivVOW-O1IAuF7rJYSms/view?usp=sharing)  
Put model file to `Course_VRDL/HW3_Instance_Segmentation/exp/model_final.pth`

## Architecture
```
Course_VRDL/
    ├── dataset/
    │   ├── train/
    │   │   ├── TCGA-18-5592-01Z-00-DX1/
    │   │   └── TCGA-21-5784-01Z-00-DX1/...
    │   ├── test/
    │   └── test_img_ids.json
    ├── mask2coco.py
    ├── testdata_coco.py
    │
    ├── exp/
    │   └── model_final.pth
    ├── try_train.py
    ├── try_valid.py
    └── vis.py
```
## Train or Test
### Testing (inference)
1. Use the following instruction to predict result
   ```
   ~/Course_VRDL$ python HW3_Instance_Segmentation/try_valid.py
   ```
   And this will output 2 files
   - `HW3_Instance_Segmentation/predict_Vis_annotation.json`  
     predict result for visualize
   - `HW3_Instance_Segmentation/predict_annotation.json`  
     predict result for codalab competition

2. Use the following instruction to Visualize result(based on `HW3_Instance_Segmentation/predict_Vis_annotation.json`)
   ```
   ~/Course_VRDL$ python HW3_Instance_Segmentation/vis.py
   ```
   And this will output predict results for each testing image in `HW3_Instance_Segmentation/dataset/test/` which is saved in `HW3_Instance_Segmentation/VisJson/`

### Training
Use the following instruction to train **mask_rcnn_R_101_FPN_3x** fine tuned on this task
```
~/Course_VRDL$ python HW3_Instance_Segmentation/try_train.py
```
The final weights will be saved in `HW3_Instance_Segmentation/exp/model_final.pth`.
Then you can use instructions in [Testing](#testing-inference) to predict result and visualize it.

## Reference (helps)
Detectron2
- https://github.com/facebookresearch/detectron2
- https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF

COCO format
- https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
- https://www.gushiciku.cn/pl/gavr/zh-tw
- https://www.aiuai.cn/aifarm1578.html

Output format
- https://github.com/facebookresearch/detectron2/issues/556#issuecomment-968035926
- https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format

Model hyper param
- https://github.com/facebookresearch/detectron2/issues/1045#issuecomment-598542491
- https://github.com/facebookresearch/detectron2/issues/277
- https://detectron2.readthedocs.io/en/latest/modules/config.html