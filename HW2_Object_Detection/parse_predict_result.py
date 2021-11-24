import os
from PIL import Image
import json
from tqdm import tqdm

ResultPath = r'../yolov5/runs/detect/exp?/labels'
TestImageFolderPath = r'HW2_Object_Detection/HW2_dataset/test_images'
DumpJsonPath = r'HW2_Object_Detection/answer.json'

def Parser(result_labels_folder, img_folder, dump_path):
    dict_collector = []
    for txt_file in tqdm(os.listdir(result_labels_folder)):
        fullpath = os.path.join(result_labels_folder, txt_file)
        img_path = os.path.join(img_folder, txt_file.split('.')[0]+'.png')
        # print(fullpath)
        image = Image.open(img_path)
        size = image.size  # (w, h)

        with open(fullpath, 'r') as file:
            for line in file:
                d = change_format(line, size, txt_file.split('.')[0])
                dict_collector += [d]
    
    with open(dump_path, 'w') as file:
        json.dump(dict_collector, file, indent=4)


def change_format(row_data: str, size: (int, int), image_id: str) -> dict:
    '''
        :param row_data: a bbox with format below
        :param size: Picture size(w, h)
        :return: dictionary with format below

        Format
        from
            'class x_center y_center width height confidence'
        to
            {
                "image_id": 117,
                "score": 0.9752130508422852,
                "category_id": 3,
                "bbox": [
                    41.071231842041016,
                    8.766018867492676,
                    13.521903991699219,
                    25.68875789642334
                ]
            }
    '''
    row_data = row_data.split()
    class_ = int(row_data[0])
    x_center = float(row_data[1])
    y_center = float(row_data[2])
    width = float(row_data[3])
    height = float(row_data[4])
    confidence = float(row_data[5])

    left = (x_center - width / 2) * size[0]
    top = (y_center - height / 2) * size[1]
    width = width * size[0]
    height = height * size[1]

    output_dict = dict()
    output_dict['image_id'] = int(image_id)
    output_dict['bbox'] = [ left, top, width, height ]
    output_dict['score'] = confidence
    output_dict['category_id'] = class_

    return output_dict

Parser(ResultPath, TestImageFolderPath, DumpJsonPath)


'''
    format form:
        [ [class x_center y_center width height conf]
        [class x_center y_center width height conf] ]
    to:
    [
        {
            "image_id": 117,
            "score": 0.9752130508422852,
            "category_id": 3,
            "bbox": [
                41.071231842041016,
                8.766018867492676,
                13.521903991699219,
                25.68875789642334
            ]
        },
        {
            "image_id": 117,
            "score": 0.07812704145908356,
            "category_id": 3,
            "bbox": [
                44.9322624206543,
                9.461458206176758,
                7.324241638183594,
                22.9411678314209
            ]
        },
    ]
'''