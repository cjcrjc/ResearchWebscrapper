from tkinter import filedialog
import matplotlib.pyplot as plt
from torchvision.models.detection import *
import numpy as np
import cv2, os
import torch
from PIL import Image
import torchvision

model = fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)
model.load_state_dict(torch.load("2023-10-16.model"))


folder = os.getcwd() + "\\Data Collection\\keep"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
for img_path in os.listdir(folder):
    model.eval()
    img = Image.open(folder+"\\"+img_path)
    image = transform(img)
    img = np.array(img)
    #print(image.shape)
    output = model([image])
    boxes = output[0]["boxes"].detach().numpy().astype(int)
    scores = output[0]["scores"].detach().numpy().astype(float)
    keepboxes = []
    for i in range(len(boxes)):
        if scores[i] > 0.6:
            keepboxes.append(boxes[i])
    boxes = keepboxes
    for i in range(len(boxes)):
        cv2.rectangle(img, tuple(boxes[i][0:2]), tuple(boxes[i][2:4]),color=(255, 0, 0), thickness=2)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.show()