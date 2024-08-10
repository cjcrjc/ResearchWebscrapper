# Import necessary libraries
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, faster_rcnn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import torchvision.transforms as transforms

# Get the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

def is_mostly_white_or_black(img, threshold=0.7):
    # Convert image to RGB
    img = img.convert("RGB")
    
    # Get image dimensions
    width, height = img.size
    total_pixels = width * height
    
    # Initialize white pixel counter
    white_pixels = 0
    black_pixels = 0
    
    # Define what counts as "white"
    white_threshold = (250, 250, 250)  # RGB values
    black_threshold = (10, 10, 10)  # RGB values

    # Count the number of white pixels
    for pixel in img.getdata():
        if pixel >= white_threshold:
            white_pixels += 1
        if pixel <= black_threshold:
            black_pixels += 1

    # Calculate the ratio of white pixels
    white_ratio = white_pixels / total_pixels
    black_ratio = black_pixels / total_pixels

    # Check if the ratio exceeds the threshold
    return white_ratio > threshold or black_ratio > threshold

# Define a function for RCNN cropping
def rcnn_crop():
    print("RCNN CROP STARTED")

    # Determine the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a Faster R-CNN model with a ResNet-50 backbone and two output classes
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2).to(device)

    # Load the pre-trained weights for different parts of the model
    model.backbone.body.load_state_dict(torch.load(os.path.join(dir_path, "PARTIALMODEL-IntermediateLayerGetter-2023-10-20.model"), map_location=device))
    model.backbone.fpn.load_state_dict(torch.load(os.path.join(dir_path, "PARTIALMODEL-FeaturePyramidNetwork-2023-10-20.model"), map_location=device))
    model.roi_heads.load_state_dict(torch.load(os.path.join(dir_path, "PARTIALMODEL-RoIHeads-2023-10-20.model"), map_location=device))
    model.rpn.load_state_dict(torch.load(os.path.join(dir_path, "PARTIALMODEL-RegionProposalNetwork-2023-10-20.model"), map_location=device))

    # Define the folder containing images to be cropped
    folder = os.path.join(os.getcwd(), "cnnsorter", "filtered")

    # Define a transformation for the image
    transform = transforms.Compose([transforms.ToTensor()])

    # Loop through images in the folder and perform cropping
    for img_path in os.listdir(folder):
        model.eval()
        img = Image.open(os.path.join(folder, img_path)).convert('RGB')
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
        save_path = os.path.join(dir_path, "cropped")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img_name = os.path.basename(img_path)
        for i in range(len(boxes)):
            # Crop subimages and save them with appropriate names
            cropped_image = img.crop(tuple(boxes[i]))
            if cropped_image.size[0] > 150 and cropped_image.size[1] > 150 and not is_mostly_white_or_black(cropped_image):
                cropped_image.save(os.path.join(save_path, img_name.replace(".", f"-{i}.")))

        # Remove the original image from the folder
        os.remove(os.path.join(folder, img_path))
        
        print(f"[+] Cropped {len(boxes)} subimages out of:", img_name)
    
    print("RCNN SORTING FINISHED")

# Entry point of the script
if __name__ == "__main__":
    rcnn_crop()