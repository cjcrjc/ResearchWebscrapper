import torch, os
from torchvision.models.detection import *
from datetime import date

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the pre-trained model
model_data = torch.load(dir_path+"/2024-08-13.model")

# Create a Fast R-CNN model with a ResNet-50 backbone and two output classes
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)

# Load the model state from the pre-trained data
model.load_state_dict(model_data)

# Iterate through the model's components and save their state dicts
for child in list(model.children()):
    if child._get_name() == "BackboneWithFPN":
        # Save the state of the backbone with FPN
        for module in child.children():
            torch.save(module.state_dict(),f'rcnn/PARTIALMODEL-{module._get_name()}-{date.today()}.model')
    elif child._get_name() == "GeneralizedRCNNTransform":
        # Skip saving the state of the transformation module
        pass
    else:
        # Save the state of other model components
        torch.save(child.state_dict(),f'rcnn/PARTIALMODEL-{child._get_name()}-{date.today()}.model')
