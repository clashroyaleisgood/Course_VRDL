'''
    move training images
        from dataset/train/*/images/*.png
        to   dataset/my_dataset/JPEGImages/*.png
    move training labels
        from dataset/annotation.json
        to   dataset/my_dataset/annotations.json

    e.g. dataset/train/TCGA-18-5592-01Z-00-DX1/images/TCGA-18-5592-01Z-00-DX1.png
      to dataset/my_dataset/JPEGImages/TCGA-18-5592-01Z-00-DX1.png

'''
import os

ReadTrainImageFolder = r'HW3_Instance_Segmentation/dataset/train'

SaveTrainDataFolder = r'HW3_Instance_Segmentation/dataset/my_dataset/train'

def MoveTrainImages():
    SaveTrainImageFolder = os.path.join(SaveTrainDataFolder, r'JPEGImages')

    PathToBuild = [
        'HW3_Instance_Segmentation/dataset/my_dataset',
        'HW3_Instance_Segmentation/dataset/my_dataset/JPEGImages'
    ]
    for path in PathToBuild:
        if not os.path.isdir(path):
            os.mkdir(path)

    ImageFolders = os.listdir(ReadTrainImageFolder)

    for img_name in ImageFolders:
        ImagePath = os.path.join(ReadTrainImageFolder, img_name, 'images', img_name + '.png')
        SavePath = os.path.join(SaveTrainImageFolder, img_name + '.png')
        os.rename(ImagePath, SavePath)
MoveTrainImages()

def MoveTrainJson():
    ReadJsonPath = r'HW3_Instance_Segmentation/dataset/annotation.json'
    SaveJsonPath = os.path.join(SaveTrainDataFolder, r'annotation.json')
    os.rename(ReadJsonPath, SaveJsonPath)
MoveTrainJson()