from PIL import Image
from tqdm import tqdm
import numpy as np
import time
import os

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ------------------------- Hyper Params            -------------------------

BatchSize = 64
NumWorkers = 4

PaddingWidth = 100
CropSize = (400, 400)

# ------------------------- PreRequirement          -------------------------

# ModelLoadPath = r'HW 1 classification/MODEL.pt'
ModelLoadPath = r'HW 1 classification/resnet50.pt'
AnswerSavePath = r'HW 1 classification/answer.txt'
if os.path.isfile(AnswerSavePath):
    answer = input(f'existing file at {AnswerSavePath}, are you sure to cover it? [y]|n')
    if answer in ['', 'n']:
        pass
    else:
        exit()

TestPath = r'HW 1 classification/2021VRDL_HW1_datasets/testing_images/'
# TestFileName = os.listdir(TestPath)
TestOrderPath = \
    r'HW 1 classification/2021VRDL_HW1_datasets/testing_img_order.txt'
AllLabelPath = r'HW 1 classification/2021VRDL_HW1_datasets/classes.txt'

# Collect Labels ( Number to String)
Labels = []  # List of Strings
with open(AllLabelPath) as file:
    for line in file:
        # line = line.split('.')[1]
        Labels += [line.strip()]
Labels = np.array(Labels)

# Collect Order of Test Image
Order = [] # List of Strings
with open(TestOrderPath) as file:
    for line in file:
        Order += [line.strip()]

test_data_size = len(Order)

# ------------------------- Dataset Preprocessing   -------------------------

preprocess = transforms.Compose([
    # transforms.AutoAugment(),  # 必須是 uint8 所以就放在 ToTensor 前
    # If the image is torch Tensor, it should be of type torch.uint8
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.Pad(PaddingWidth),
    transforms.CenterCrop(CropSize),
    # transforms.Resize((375, 500)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

# ------------------------- Dataset / DataLoader    -------------------------

class MyDataset(Dataset):
    def __init__(self, filenames, folderpath, transform):
        self.filenames = filenames  # ['1.jpg'

        self.folderpath = folderpath
        self.transform = transform
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(self.folderpath + filename)  # check dimension

        if self.transform:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.filenames)

test_dataset = MyDataset(Order, TestPath, preprocess)
test_data_loader = DataLoader(dataset=test_dataset,
                               batch_size=BatchSize,
                               shuffle=False,
                               num_workers=NumWorkers)

# ------------------------- Choose Model            -------------------------

def GetModel(read_model_path=None):
    model = None
    if read_model_path and os.path.isfile(read_model_path):
        print(f'Load Model from {read_model_path}')
        model = torch.load(read_model_path)
    else:
        print(f'ERROR: can\'t find model at path {read_model_path}')

    return model

wide_resnet50_2 = GetModel(ModelLoadPath)
wide_resnet50_2 = wide_resnet50_2.to('cuda')

# ------------------------- Predict                 -------------------------

def ModelPredict(model, test_data):
    '''
    model: .pt model
    test_data: DataLoader()

    return: [int, int, int, ..., int]
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'to device: {device}')
    predict_results = []

    how_many_batches = np.ceil(test_data_size / BatchSize)
    for inputs in tqdm(test_data, total=how_many_batches):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        predict_results += [predictions.cpu().numpy()]

    predict_results = np.concatenate(predict_results)
    return np.array(predict_results).flatten()

predict_order = ModelPredict(
    model=wide_resnet50_2,
    test_data=test_data_loader
)
predict_results = Labels[predict_order]


submission = []
for i in range(test_data_size):
    submission.append([Order[i], predict_results[i]])

np.savetxt(AnswerSavePath, submission, fmt='%s')


