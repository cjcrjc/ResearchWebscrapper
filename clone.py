import os
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from PIL import Image
import numpy as np

check_path=  r'C:\Users\Cameron\Desktop\Coding\Python\unsuperShapelets\cropped'
all_path = r'C:\Users\Cameron\Desktop\Coding\Python\unsuperShapelets\production\classify\data'

current = []
new = []
for set in os.walk(all_path):
    for sublist in set:
        if type(sublist) == list:
            for file in sublist:
                if len(file) > 10:
                    current.append(os.path.join(set[0], file))

for set in os.walk(check_path):
    for sublist in set:
        if type(sublist) == list:
            for file in sublist:
                if len(file) > 10:
                    new.append(os.path.join(check_path, file))

for file in new:
    mses = []
    ssims = []
    convs = []
    convs2 = []
    img1 = np.array(Image.open(file).convert("L"))
    img1 = (img1 - np.mean(img1))
    for file2 in current:
        img2 = np.array(Image.open(file2).convert("L").resize((img1.shape[1], img1.shape[0])))
        img2_unsized = np.array(Image.open(file2).convert("L"))
        img2_unsized = (img2_unsized - np.mean(img2_unsized))
        pad_width = ((0, img1.shape[0] - img2_unsized.shape[0]), (0, img1.shape[1] - img2_unsized.shape[1]))
        if img1.shape[0] > img2_unsized.shape[0] and img1.shape[1] > img2_unsized.shape[1]:
            pad_height = img1.shape[0] - img2_unsized.shape[0]
            pad_width = img1.shape[1] - img2_unsized.shape[1]
            img2_padded = np.pad(img2_unsized, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=np.mean(img2_unsized))
            mses.append(mse(img1, img2))
            ssims.append(ssim(img1, img2_padded, data_range=img1.max()-img1.min()))
    if len(mses) > 1:
        print(min(mses), file)