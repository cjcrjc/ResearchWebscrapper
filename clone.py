import os
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from time import time

check_path=  r'C:\Users\Cameron\Desktop\Coding\Python\unsuperShapelets\new'
all_path = r'C:\Users\Cameron\Desktop\Coding\Python\unsuperShapelets\production\classify\data'

current = []
new = []
for set in os.walk(all_path):
    for sublist in set:
        if type(sublist) == list:
            for file in sublist:
                if len(file) > 10:
                    current.append(file)

for set in os.walk(check_path):
    for sublist in set:
        if type(sublist) == list:
            for file in sublist:
                if len(file) > 10:
                    new.append(file)

for file in new:
    for file2 in current:
        if file[:10] == file2[:10]:
            print(file)
            break