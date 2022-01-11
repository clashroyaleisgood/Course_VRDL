# Homework4 - Super Resolution

You can see report [here](report/report.md) in markdown format, or [pdf](report/VRDL_HW4_309553018_Report.pdf) format which is exported by [Typora](https://typora.io/).

Competition: https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400

Result score on testing dataset: **27.4265**

Model weight: [best.pth](https://drive.google.com/file/d/1R4Vjkuz-_aKssFjzoG9PfsJKE1yAUe0Z/view?usp=sharing)

## Table of Contents

- [Homework4 - Super Resolution](#homework4---super-resolution)
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

## Environment
Install CUDA, cuDNN, PyTorch first  
modules:
```
$ pip install -r requirements.txt
```
## Code
1. Download code with the following command:
`$ git clone https://github.com/clashroyaleisgood/Course_VRDL.git`
2. Download ESPCN with the following command:
`$ git clone https://github.com/yjn870/ESPCN-pytorch.git`
3. Copy files in `Course_VRDL/HW4_Super_Resolution/code/` to `ESPCN-pytorch/`
```
Desktop/
    ├── Course_VRDL/HW4_Super_Resolution/code
    │   ├── test.py  # copy form here
    │   └── train.py  # and here
    └── ESPCN-pytorch/
        ├── test.py  #      to here
        └── train.py  # and here
```

## Dataset
Get data from this [competition](https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400), or this [Dataset Link](https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view?usp=sharing)

Put Training data to `HW4_Super_Resolution/datasets/`
```
Course_VRDL/HW4_Super_Resolution/
    └── datasets/
        ├── testing_lr_images/testing_lr_images/
        │   ├── 00.png
        │   └── 01.png...
        └── training_hr_images/training_hr_images/
            └── ...
```

## Preprocessing
1. execute `~/Course_VRDL$ python HW4_Super_Resolution/split_validation.py` to split training / validation data by 9:1 ratio
2. Copy `Course_VRDL/HW4_Super_Resolution/datasets/` folder to `ESPCN-pytorch/datasets/`


## Model
Get weight with this [link]()  
Put model file to `ESPCN-pytorch/output/3x/best.pth`

## Architecture
```
Desktop/
    ├── Course_VRDL/HW4_Super_Resolution/
    │   ├── datasets/
    │   ├── code
    │   ├── test.py
    │   └── train.py
    │
    └── ESPCN-pytorch/
        ├── datasets/
        │   ├── testing_lr_images/testing_lr_images/
        │   │   ├── 00.png
        │   │   └── 01.png...
        │   ├── training_hr_images/training_hr_images/
        │   │   └── ...
        │   └── training_hr_images/training_hr_images/
        │       └── ...
        │
        ├── outputs/x3/
        │   └── best.pth
        │
        ├── prepare.py
        ├── test.py
        ├── train.py
        └── ...
```
## Train or Test
### Testing (inference)
execute
```
ESPCN-pytorch/$ python test.py \
	--weights-file "outputs/x3/best.pth" \
	--folder "datasets/testing_lr_images/testing_lr_images" \
	--infer \
	--scale 3
```
Then the upscaled result will be stored in `ESPCN-pytorch/datasets/testing_lr_images/testing_lr_images/3x/`

### Training
Execute
```
ESPCN-pytorch/$ python train.py \
	--train-file "datasets/train_l.h5" \
	--eval-file "datasets/valid_l.h5" \
	--outputs-dir "outputs" \
	--scale 3 \
	--lr 1e-3 \
	--batch-size 8 \
	--num-epochs 300 \
	--num-workers 16 \
	--seed 123
```
Then training weights will be stored in `ESPCN-pytorch/outputs/x3/`
