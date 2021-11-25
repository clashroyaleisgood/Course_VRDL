'''
reference: https://www.vitaarca.net/post/tech/access_svhn_data_in_python/
'''
import h5py
import os
from tqdm import tqdm
from PIL import Image

MatFilePath = r'HW2_Object_Detection/HW2_dataset/train_images/digitStruct.mat'
PicturePath = r'HW2_Object_Detection/HW2_dataset/train_images'

TrainImagePath = r'HW2_Object_Detection/HW2_dataset/images/train'
TrainLabelPath = r'HW2_Object_Detection/HW2_dataset/labels/train'

ValidImagePath = r'HW2_Object_Detection/HW2_dataset/images/valid'
ValidLabelPath = r'HW2_Object_Detection/HW2_dataset/labels/valid'

# Picture with too LARGE bbox size
GarbageImagePath = r'HW2_Object_Detection/HW2_dataset/images/garbage'

# -----------------------------------------------------------------------------
bbox_prop = ['height', 'left', 'top', 'width', 'label']

def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)

def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label'
        of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    meta = {key: [] for key in bbox_prop}

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta

def change_format(meta, size):
    """
    translate format from {'height':[, ], 'left':[, ], 'top':[, ],
                           'width':[, ], 'label':[, ]}
                       to [ [class, x_center, y_center, width, height]
                            [class, x_center, y_center, width, height] ]
    :param meta: output of get_img_boxes()
    :param size: (width, height) of picture

    :return: list of [class, x_center, y_center, width, height]

    """
    bbox_numbers = len(meta['label'])
    rows = []
    for i in range(bbox_numbers):
        class_ = meta['label'][i]
        x_center = (meta['left'][i] + meta['width'][i] / 2) / size[0]
        y_center = (meta['top'][i] + meta['height'][i] / 2) / size[1]
        width = meta['width'][i] / size[0]
        height = meta['height'][i] / size[1]

        if class_ == 10:
            class_ = 0
        if x_center + width / 2 > 1:
            print(f'[ERROR] box width too large')
            print(f'size: {size}')
            print(f'left: {meta["left"][i]}, width: {meta["width"][i]}')
            return False
        if y_center + height / 2 > 1:
            print(f'[ERROR] box height too large')
            print(f'size: {size}')
            print(f'top: {meta["top"][i]}, height: {meta["height"][i]}')
            return False

        rows.append([class_, x_center, y_center, width, height])

    return rows

# -----------------------------------------------------------------------------
f = h5py.File(MatFilePath, 'r')
names = f['digitStruct/name']
bboxs = f['digitStruct/bbox']

image_size = names.shape[0]
train_size = int(image_size * 0.8)
valid_size = image_size - train_size

print(f'Preprocessing {image_size} training images')
print(f'{train_size} images for training, {valid_size} images for validation')
# -----------------------------------------------------------------------------
PathToBuild = [
    'HW2_Object_Detection/HW2_dataset/images',
    'HW2_Object_Detection/HW2_dataset/images/train',
    'HW2_Object_Detection/HW2_dataset/images/valid',
    'HW2_Object_Detection/HW2_dataset/images/garbage',
    'HW2_Object_Detection/HW2_dataset/labels',
    'HW2_Object_Detection/HW2_dataset/labels/train',
    'HW2_Object_Detection/HW2_dataset/labels/valid'
]
print('\nBuilding Path:')
print(PathToBuild, sep='\n')
for path in PathToBuild:
    if not os.path.isdir(path):
        os.mkdir(path)
# -----------------------------------------------------------------------------

print(f'\n{train_size} training images')
print(f'Moving {train_size} pictures from {PicturePath}/ to {TrainImagePath}/')
print(f'Write label files to {TrainLabelPath}/')

for i in tqdm(range(0, train_size)):
    pic_name = get_img_name(f, i)
    picture = Image.open(f'{PicturePath}/{pic_name}')

    pic_meta = get_img_boxes(f, i)
    row_data = change_format(pic_meta, picture.size)
    if row_data is False:
        os.rename(f'{PicturePath}/{pic_name}',
                  f'{GarbageImagePath}/{pic_name}')
        continue

    # 1. move picture  2. write lable.txt
    os.rename(f'{PicturePath}/{pic_name}', f'{TrainImagePath}/{pic_name}')
    with open(f'{TrainLabelPath}/{pic_name.split(".")[0]}.txt', 'w') \
            as label_file:
        for row in row_data:
            # label_file.write(''.join(str(e) for e in row) + '\n')
            print(*row, sep=' ', end='\n', file=label_file)

print(f'\n{valid_size} validation images')
print(f'Moving {valid_size} pictures from {PicturePath}/ to {ValidImagePath}/')
print(f'Write label files to {ValidLabelPath}/')

for i in tqdm(range(train_size, image_size)):
    pic_name = get_img_name(f, i)
    picture = Image.open(f'{PicturePath}/{pic_name}')

    pic_meta = get_img_boxes(f, i)
    row_data = change_format(pic_meta, picture.size)
    if row_data is False:
        os.rename(f'{PicturePath}/{pic_name}',
                  f'{GarbageImagePath}/{pic_name}')
        continue

    # 1. move picture  2. write lable.txt
    os.rename(f'{PicturePath}/{pic_name}', f'{ValidImagePath}/{pic_name}')
    with open(f'{ValidLabelPath}/{pic_name.split(".")[0]}.txt', 'w') \
            as label_file:
        for row in row_data:
            # label_file.write(''.join(str(e) for e in row) + '\n')
            print(*row, sep=' ', end='\n', file=label_file)
