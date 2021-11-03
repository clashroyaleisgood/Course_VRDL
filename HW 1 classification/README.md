# Homework1 - Classification

## Environment
Install CUDA, cuDNN, PyTorch

```python=
pip install numpy matplotlib tqdm scikit-learn
pip install PyQt5  # (Optional) Ubuntu only
sudo apt-get install libxcb-xinerama0  # (Optional) If
    ERROR: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    happend.
```

## Dataset
get data from this [competetion](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07)
- Train Data  
  put Training data to `HW 1 classification/training_images/`
- Test Data  
  put Training data to `HW 1 classification/testing_images/`

like
```
HW 1 classification/
        ├─ HW 1.py
        ├─ HW 1 Predict.py
        └─ submission_readme/
                └─ ...
        └─ 2021VRDL_HW1_datasets/
                ├─ classes.txt
                ├─ testing_img_order.txt
                ├─ training_labels.txt
                └─ training_images/
                        │─ 0003.jpg
                │       │─ 0008.jpg
                │       └─ ...
                └─ testing_images/
                        │─ 0001.jpg
                        └─ ...
```
by: https://zh.wikipedia.org/wiki/%E6%96%B9%E6%A1%86%E7%BB%98%E5%88%B6%E5%AD%97%E7%AC%A6

