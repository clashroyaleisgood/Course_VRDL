'''
must be executed AFTER downscale.py

datasets/
    ├── training_hr_images/training_hr_images/
    ├── training_lr_images/training_lr_images/
    │   ├── train00.png
    │   └── train01.png
    ├── validate_hr_images/validate_hr_images/
    └── validate_lr_images/validate_lr_images/
        ├── valid00.png
        └── valid01.png
'''

import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

VALID_SIZE = 0.1
RAND_SEED = 123

train_hr_path = r'HW4_Super_Resolution/datasets/training_hr_images/training_hr_images'
train_lr_path = r'HW4_Super_Resolution/datasets/training_lr_images/training_lr_images'
valid_hr_path_store = r'HW4_Super_Resolution/datasets/validate_hr_images/validate_hr_images'
valid_lr_path_store = r'HW4_Super_Resolution/datasets/validate_lr_images/validate_lr_images'

os.makedirs(valid_hr_path_store, exist_ok=True)
os.makedirs(valid_lr_path_store, exist_ok=True)

image_names = os.listdir(train_hr_path)

train_name, valid_name = train_test_split(image_names, test_size=VALID_SIZE, random_state=RAND_SEED)

for names in tqdm(valid_name):
    hr_from_path = os.path.join(train_hr_path, names)
    lr_from_path = os.path.join(train_lr_path, names)
    hr_to_path = os.path.join(valid_hr_path_store, names)
    lr_to_path = os.path.join(valid_lr_path_store, names)

    os.rename(hr_from_path, hr_to_path)
    os.rename(lr_from_path, lr_to_path)
