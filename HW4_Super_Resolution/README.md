# Homework4 - Super Resolution

You can see report [here](report/report.md) in markdown format, or [pdf](report/VRDL_HW4_309553018_Report.pdf) format which is exported by [Typora](https://typora.io/).

Competition: https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400

Result score on testing dataset: **???**

Model weight: [model_final.pth]()

## Table of Contents

- [Homework4 - Super Resolution](#homework4---super-resolution)
  - [Table of Contents](#table-of-contents)
  - [Environment](#environment)
  - [Code](#code)
  - [Dataset](#dataset)
      - [TODO](#todo)
  - [Preprocessing](#preprocessing)
  - [Model](#model)
  - [Architecture](#architecture)
  - [Train or Test](#train-or-test)
    - [Testing (inference)](#testing-inference)
    - [Training](#training)
  - [Reference (helps)](#reference-helps)

## Environment

## Code
Download code with the following command:  
`$ git clone https://github.com/clashroyaleisgood/Course_VRDL.git`

## Dataset
Get data from this [competition](https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400), or this [Dataset Link](https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view)

Put Training data to `HW4_Super_Resolution/datasets/`
#### TODO
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


## Model
Get weight with this [link]()  
Put model file to `Course_VRDL/`

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


### Training


## Reference (helps)
