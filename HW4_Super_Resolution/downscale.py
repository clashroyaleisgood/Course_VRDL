import cv2
import os
from tqdm import tqdm

train_hr_path = r'HW4_Super_Resolution/datasets/training_hr_images/training_hr_images'
train_lr_store_path = r'HW4_Super_Resolution/datasets/training_lr_images/training_lr_images'
os.makedirs(train_lr_store_path, exist_ok=True)

image_names = os.listdir(train_hr_path)

for name in tqdm(image_names):
    full_path = os.path.join(train_hr_path, name)
    hr_image = cv2.imread(full_path)
    # height, width, _ = hr_image.shape

    lr_image = cv2.resize(hr_image, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(train_lr_store_path, name), lr_image)
