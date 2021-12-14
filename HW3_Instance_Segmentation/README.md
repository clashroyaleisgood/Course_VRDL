# Homework3 - Instance Segmentation

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
    - [Testing](#testing)
    - [Training](#training)
  - [Result](#result)

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
Get data from this [competetion](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5)
, [Dataset Link](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view?usp=sharing)

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
Get weight with this [link]()  
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
    │    └── model_final.pth
    ├── try_train.py
    ├── try_valid.py
    └── vis.py
```
## Train or Test
### Testing

### Training


## Result