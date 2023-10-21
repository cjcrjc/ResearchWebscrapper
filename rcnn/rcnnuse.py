# Import necessary libraries
from torchvision.models.detection import *
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch
from PIL import Image
import torchvision

# Get the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define a function for RCNN cropping
def rcnn_crop():
    # Determine the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a Faster R-CNN model with a ResNet-50 backbone and two output classes
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2).to(device)

    # Load the pre-trained weights for different parts of the model
    model.backbone.body.load_state_dict(torch.load(dir_path + "/PARTIALMODEL-IntermediateLayerGetter-2023-10-20.model", device))
    model.backbone.fpn.load_state_dict(torch.load(dir_path + "/PARTIALMODEL-FeaturePyramidNetwork-2023-10-20.model", device))
    model.roi_heads.load_state_dict(torch.load(dir_path + "/PARTIALMODEL-RoIHeads-2023-10-20.model", device))
    model.rpn.load_state_dict(torch.load(dir_path + "/PARTIALMODEL-RegionProposalNetwork-2023-10-20.model", device))

    # Define the folder containing images to be cropped
    folder = os.getcwd() + "/cnnsorter/filtered"

    # Define a transformation for the image
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Loop through images in the folder and perform cropping
    for img_path in os.listdir(folder):
        model.eval()
        img = Image.open(folder + "/" + img_path).convert('RGB')
        image = transform(img).to(device)
        output = model([image])
        boxes = output[0]["boxes"].detach().cpu().numpy().astype(int)
        scores = output[0]["scores"].detach().cpu().numpy().astype(float)
        keepboxes = []
        
        # Filter boxes based on confidence scores
        for i in range(len(boxes)):
            if scores[i] > 0.6:
                keepboxes.append(boxes[i])
        boxes = keepboxes

        # Define the save path for cropped images
        save_path = dir_path + "/cropped"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_name = os.path.basename(img_path)
        for i in range(len(boxes)):
            # Crop subimages and save them with appropriate names
            cropped_image = img.crop(tuple(boxes[i]))
            cropped_image.save(os.path.join(dir_path, "cropped", img_name.replace(".", f"-{i}.")))

        # Remove the original image from the folder
        os.remove(folder + "/" + img_path)
        
        print(f"[+] Cropped {len(boxes)} subimages out of:", os.path.basename(img_path))
    
    print("RCNN SORTING FINISHED")
