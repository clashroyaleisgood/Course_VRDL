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
from torchvision import models


# ------------------------- Hyper Params            -------------------------
# Hyper for spliting data to train / valid

# 3000 images
ValidationPercentage = 0.2  # 0.8: train, 0.2: validation, 2400:600
RandomSeed = 99  # for train_test_split()

# -----

BatchSize = 16
EpochCounts = 30
NumWorkers = 4

LearningRate = 0.01
WeightDecay = 0.001  # L2 loss
OptimizerType = lambda params: \
    optim.SGD(params, lr=LearningRate, weight_decay=WeightDecay, momentum=0.9)
OptimizerString = f'SGD - lr: {LearningRate}, WD: {WeightDecay}, momenton: 0.9'

# SchedulerType = lambda opter: \
#     optim.lr_scheduler.MultiStepLR(opter, milestones=[5, 10, 20], gamma=0.2)
# SchedulerString = r"MultiStepLR - milestones=[5, 10, 20], γ=0.2"
# SchedulerType = lambda opter: optim.lr_scheduler.ReduceLROnPlateau(opter)
# SchedulerString = r'ReduceLROnPlateau'
SchedulerType = lambda opter: None
SchedulerString = 'None'

PaddingWidth = 100
CropSize = (400, 400)

DropoutRate = 0.5  # Architecture: Dropout layer rate, end of model

# ------------------------- PreRequirement          -------------------------
# Data Path
TrainPath = r'HW 1 classification/2021VRDL_HW1_datasets/training_images/'
TestPath = r'HW 1 classification/2021VRDL_HW1_datasets/testing_images/'
TrainFileName = os.listdir(TrainPath)  # ['1.jpg', '5.jpg', ...]
TestFileName = os.listdir(TestPath)
# TrainSize = len(TrainFileName)
TrainLabelPath = \
    r'HW 1 classification/2021VRDL_HW1_datasets/training_labels.txt'
AllLabelPath = r'HW 1 classification/2021VRDL_HW1_datasets/classes.txt'

# Model Path
ModelLoadPath = r'HW 1 classification/wide_resnet50_2.pt'
ModelSavePath = r'HW 1 classification/wide_resnet50_2.pt'

# History Path
HistorySavePath = r'HW 1 classification/history.npy'

# Collect Labels
Labels = []  # List of Strings
with open(AllLabelPath) as file:
    for line in file:
        # line = line.split('.')[1]
        Labels += [line.strip()]

# Split Image Name by Classes
# Use train_test_split
def CollectTrainingData(train_data_label_path):
    '''
    return  X_train: ['1.jpg', '2.jpg', ...]
            X_validation: same as above...
            y_train: [0, 4, 12, 199, ...]  # 001.Black_footed_Albatross -> 0
            y_validation: same as above...
    '''
    X = []
    y = []
    with open(train_data_label_path) as file:

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

X_train, X_valid, y_train, y_valid = CollectTrainingData(TrainLabelPath)
train_data_size = len(X_train)
valid_data_size = len(X_valid)

# ------------------------- Dataset Preprocessing   -------------------------

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.Pad(PaddingWidth),
    transforms.CenterCrop(CropSize),
    transforms.RandomHorizontalFlip(p=0.5),
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
        model = models.wide_resnet50_2(pretrained=True)

        for name, child in model.named_children():
            if name in ['layer4', 'avgpool', 'fc']:
                print(f'{name} is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                print(f'{name} is frozen')
                for param in child.parameters():
                    param.requires_grad = False

        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DropoutRate),
            nn.Linear(fc_inputs, 200),
            nn.LogSoftmax(dim=1)
        )
    return model

wide_resnet50_2 = GetModel()
# wide_resnet50_2 = GetModel(ModelLoadPath)
wide_resnet50_2 = wide_resnet50_2.to('cuda')

# ------------------------- Train / Test Functions  -------------------------
# optimizer = optim.Adam(
#     wide_resnet50_2.parameters(), lr=LearningRate, weight_decay=WeightDecay)
optimizer = OptimizerType(wide_resnet50_2.parameters())
scheduler = SchedulerType(optimizer)

def TrainModel(model, train_data, valid_data,
               loss_function, optimizer, scheduler, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'to device: {device}')
    history = []
    lowest_valid_loss = 100.0

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        model.train()

        # -------------------------------------------------------------------

        # Calculate Training Loss
        train_loss = 0.0
        train_acc = 0.0

        # for input, label in tqdm(train_data):
        how_many_batches = np.ceil(train_data_size / BatchSize)
        for inputs, labels in tqdm(train_data, total=how_many_batches):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因為這裡的梯度是累加的，所以每次記得清零
            optimizer.zero_grad()

            outputs = model(inputs)

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

        how_many_batches = np.ceil(valid_data_size / BatchSize)
        for inputs, labels in tqdm(valid_data, total=how_many_batches):

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

        history.append([avg_train_loss, avg_train_acc,
                        avg_valid_loss, avg_valid_acc])

        print(f'Epoch: {epoch+1}, Loss: {avg_train_loss:.4f}, ' +
              f'Accuracy: {avg_train_acc*100:.2f}%', end='')
        print(f' - Validation, Loss: {avg_valid_loss:.4f}, ' +
              f'Accuracy: {avg_valid_acc*100:.2f}%')

        if lowest_valid_loss > history[-1][2]:
            lowest_valid_loss = history[-1][2]
            torch.save(model, ModelSavePath)

        if scheduler:
            scheduler.step()
            print(scheduler.get_last_lr()[0])

    return model, history

trained_model, history = TrainModel(
    model=wide_resnet50_2,
    train_data=train_data_loader,
    valid_data=valid_data_loader,
    loss_function=nn.NLLLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=EpochCounts
)

torch.save(trained_model, ModelSavePath)

# ------------------------- Display / Save history  -------------------------

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

    fig, ax = plt.subplots()
    twins = ax.twinx()
    twins.set_ylim(0, 1)
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    twins.set_ylabel('Accuracy')

    # plt.yscale('log')
    p1, = ax.plot(train_L, 'b', label='train_L')
    p2, = twins.plot(train_A, 'c', label='train_A')
    p3, = ax.plot(valid_L, 'g', label='valid_L')
    p4, = twins.plot(valid_A, 'y', label='valid_A')

    plt.title(f'Bth Sz: {BatchSize} Drop: {DropoutRate}' +
              f'Optimizer: {OptimizerString}\n' +
              f'wide_resnet50_2 ' +
              f'Scheduler: {SchedulerString}')
    # plt.legend()
    ax.legend(handles=[p1, p2, p3, p4])
    fig.savefig('HW 1 classification/history.jpg')
    plt.show()

DisplayResult(history)
np.save(HistorySavePath, np.array(history))
