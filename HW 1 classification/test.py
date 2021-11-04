from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# with open('2021VRDL_HW1_datasets/training_labels.txt') as file:
#     for f, classes in file:
#         print(f, classes)

def gene():
    a = 0
    while a < 10:
        yield a
        a += 1

for e in tqdm(gene(), total=10.2):
    time.sleep(0.5)


'''
a = np.array([
    '[1, 2]',
    '[5, 6]',
    '[3, 4]',
    '[7, 8]'
])
b = ['1 * 1', '2 * 2', '1 * 1', '2 * 2']

print(train_test_split(a, b, test_size=0.5, stratify=b))
'''

'''
a = [
    [1,2],[3,4],[5,6],[7,8],[9,10]
]
a = np.array(a)
a = a.T
tl, vl = a

plt.plot(tl, label='train loss')
plt.plot(vl, label='valid loss')
plt.legend()
plt.show()
'''
