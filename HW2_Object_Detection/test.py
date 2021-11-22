'''

python train.py \
    --img 320 \
    --batch 32 \
    --epochs 10 \
    --data ../Course_VRDL/HW2_Object_Detection/HW2_dataset/dataset.yaml \
    --weights yolov5m.pt \
    --freeze 20

python detect.py \
    --weights runs/train/exp11/weights/best.pt \
    --source ../Course_VRDL/HW2_Object_Detection/HW2_dataset/test_images \
    --save-txt \
    --save-conf

'''