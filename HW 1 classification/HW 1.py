from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch import nn, optim
# from torch import optim
from torchvision import models


# ------------------------- Hyper Params            -------------------------
# Hyper for spliting data to train / valid

# 3000 images
ValidationPercentage = 0.2  # 0.8: train, 0.2: validation, 2400:600
RandomSeed = 99  # for train_test_split()

# -----

BatchSize = 300
EpochCounts = 10
NumWorkers = 2

LearningRate = 0.001

ClassCount = 200

# ------------------------- PreRequirement          -------------------------
# Data Path
TrainPath = r'2021VRDL_HW1_datasets/training_images/'
TestPath = r'2021VRDL_HW1_datasets/testing_images/'
TrainFileName = os.listdir(TrainPath)  # ['1.jpg', '5.jpg', ...]
TestFileName = os.listdir(TestPath)
# TrainSize = len(TrainFileName)

# Model Path
ModelLoadPath = r'resnet152.pt'
ModelSavePath = r'resnet152.pt'

# Collect Labels
Labels = []  # List of Strings
with open('2021VRDL_HW1_datasets/classes.txt') as file:
    for line in file:
        # line = line.split('.')[1]
        Labels += [line.strip()]

# Split Image Name by Classes
# Use train_test_split
def CollectTrainingData():
    '''
    return  X_train: ['1.jpg', '2.jpg', ...]
            X_validation: same as above...
            y_train: [0, 4, 12, 199, ...]  # 001.Black_footed_Albatross -> 0
            y_validation: same as above...
    '''
    X = []
    y = []
    with open('2021VRDL_HW1_datasets/training_labels.txt') as file:

        for line in file:
            image, classes = line.split()
            X += [image]
            classes = int(classes.split('.')[0]) - 1
            y += [classes]
            # print(classes.split('.')[0])

            # FileNameByClass[classes] += [image]
    return train_test_split(X, y, stratify=y,
                            test_size=ValidationPercentage,
                            random_state=RandomSeed)

X_train, X_valid, y_train, y_valid = CollectTrainingData()
train_data_size = len(X_train)
valid_data_size = len(X_valid)

# ------------------------- Dataset Preprocessing   -------------------------

preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
def TestPreprocessing(img_path, preprocess):
    '''
    img_path: image full path
    preprocess: transform.Compose([bla, blabla, ..., bla])
    '''
    img = Image.open(img_path)
    img.show()  # original
    # -----------
    img = preprocess(img)
    img = transforms.ToPILImage()(img)  # ToImg = transforms.ToPILImage()
    img.show()  # after preprocess

    # data = np.array(img)
    # plt.imshow(img.permute(1, 2, 0))
    # plt.show()

# TestPreprocessing(TrainPath + TrainFileName[0], preprocess)

# ------------------------- Dataset / DataLoader    -------------------------

class MyDataset(Dataset):
    def __init__(self, filenames, labels, folderpath, transform):
        self.filenames = filenames  # ['1.jpg'
        self.labels = labels        # [0

        self.folderpath = folderpath
        self.transform = transform
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(self.folderpath + filename)  # check dimension
        # img = np.array(img)

        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[index]

    def __len__(self):
        return len(self.filenames)

train_dataset = MyDataset(X_train, y_train, TrainPath, preprocess)
train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=BatchSize,
                               shuffle=True,
                               num_workers=NumWorkers)

valid_dataset = MyDataset(X_valid, y_valid, TrainPath, preprocess)
valid_data_loader = DataLoader(dataset=valid_dataset,
                               batch_size=BatchSize,
                               shuffle=True,
                               num_workers=NumWorkers)

# ------------------------- Build / Choose Model    -------------------------

def GetModel(read_model_path=None):
    model = None
    if read_model_path and os.path.isfile(read_model_path):
        print(f'Load Model from {read_model_path}')
        model = torch.load(read_model_path)
    else:
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 200),
            nn.LogSoftmax(dim=1)
        )
    return model

resnet152 = GetModel()
# resnet152 = GetModel(ModelLoadPath)
resnet152 = resnet152.to('cuda')

# ------------------------- Train / Test Functions  -------------------------

optimizer = optim.Adam(resnet152.parameters(), lr=LearningRate)
print()

def TrainModel(model, train_data, valid_data, loss_function, optimizer, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'to device: {device}')
    history = []
 
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
 
        model.train()

        # -------------------------------------------------------------------

        # Calculate Training Loss
        train_loss = 0.0
        train_acc = 0.0

        # for input, label in tqdm(train_data):
        how_many_batches = train_data_size / BatchSize
        for inputs, labels in tqdm(train_data, total=how_many_batches):
            # if i % 2 == 0:
            #     print(f'Train - Epoch: {epoch+1}/{epochs}, Batch: {i}/{how_many_batches}')
            inputs = inputs.to(device)
            labels = labels.to(device)
 
            # 因為這裡的梯度是累加的，所以每次記得清零
            optimizer.zero_grad()

            outputs = model(inputs)
            # outputs = torch.argmax(outputs, 1) # one hot to normal encoding
            # outputs = outputs.view([-1, 1])
            # outputs = outputs.to(torch.int64)

            # labels = labels.to(torch.int64)
            # labels = labels.squeeze()

            loss = loss_function(outputs, labels)  # 數字即可，不用 one-hot
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)
            
            # - Accuracy -
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            train_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # -------------------------------------------------------------------

        # Calculate Validation Loss
        valid_loss = 0.0
        valid_acc = 0.0

        how_many_batches = valid_data_size / BatchSize
        for inputs, labels in tqdm(valid_data, total=how_many_batches):
            # if i % 2 == 0:
            #     print(f'Valid - Epoch: {epoch+1}/{epochs}, Batch: {i}/{how_many_batches}')
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)

            # - Accuracy -
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            valid_acc += acc.item() * inputs.size(0)

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        # -------------------------------------------------------------------

        # history.append([avg_train_loss, avg_train_acc])
        history.append([avg_train_loss, avg_train_acc,
                        avg_valid_loss, avg_valid_acc])

        print(f'Epoch: {epoch+1}, Loss: {avg_train_loss:.4f}, '+ \
            'Accuracy: {avg_train_acc*100:.2f}%', end='')
        print(f' - Validation, Loss: {avg_valid_loss:.4f}, '+ \
                  'Accuracy: {avg_valid_acc*100:.2f}%')

        # torch.save(model, ModelSavePath)
        if history[-1][2] > history[-2][2]:
            torch.save(model, ModelSavePath.split('.')[0]+f'_Epoch_{epoch+1}.pt')
    
    return model, history

trained_model, history = TrainModel(
    model=resnet152,
    train_data=train_data_loader,
    valid_data=valid_data_loader,
    loss_function=nn.NLLLoss(),
    optimizer=optimizer,
    epochs=EpochCounts
)

torch.save(trained_model, ModelSavePath)

# ------------------------- Train / Test Functions  -------------------------

def DisplayResult(history):
    '''
    history:
    [
        [train_loss, train_acc, valid_loss, valid_acc],
        [],
    ]
    '''
    history = np.array(history)
    history = history.T
    train_L, train_A, valid_L, valid_A = history

    plt.plot(train_L, label='train_L')
    plt.plot(train_A, label='train_A')
    plt.plot(valid_L, label='valid_L')
    plt.plot(valid_A, label='valid_A')
    plt.title('Training')
    plt.legend()
    plt.show()

DisplayResult(history)