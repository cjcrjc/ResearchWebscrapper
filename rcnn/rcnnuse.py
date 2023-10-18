from tkinter import filedialog
import matplotlib.pyplot as plt
from torchvision.models.detection import *
import numpy as np
import cv2, os
import torch
from PIL import Image
import torchvision
dir_path = os.path.dirname(os.path.realpath(__file__))

def rcnn_crop():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(dir_path+"/2023-10-17.model",device))

    folder = os.getcwd() + "/cnnsorter/filtered"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    for img_path in os.listdir(folder):
        model.eval()
        img = Image.open(folder+"/"+img_path).convert('RGB')
        image = transform(img)
        output = model([image])
        boxes = output[0]["boxes"].detach().numpy().astype(int)
        scores = output[0]["scores"].detach().numpy().astype(float)
        keepboxes = []
        for i in range(len(boxes)):
            if scores[i] > 0.6:
                keepboxes.append(boxes[i])
        boxes = keepboxes

        #Plotting
        # img_np = np.array(img_np)
        # for i in range(len(boxes)):
        #     cv2.rectangle(img_np, tuple(boxes[i][0:2]), tuple(boxes[i][2:4]),color=(255, 0, 0), thickness=2)
        # plt.figure(figsize=(20,30))
        # plt.imshow(img_np)
        # plt.show()

        save_path = dir_path + "/cropped"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        img_name = os.path.basename(img_path)
        for i in range(len(boxes)):
            cropped_image = img.crop(tuple(boxes[i]))
            cropped_image.save(os.path.join(dir_path, "cropped", img_name.replace(".", f"-{i}.")))
        print(f"[+] Cropped {i} subimages out of:", os.path.basename(img_path))
        
        os.remove(folder+"/"+img_path)