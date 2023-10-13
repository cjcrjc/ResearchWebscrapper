from sortercnn import *
from PIL import Image
import matplotlib.pyplot as plt


rpn = RPN(in_channels=3, num_anchors=9)

image = transformer(Image.open("C:\\Users\\Cameron\\Desktop\\Coding\\Projects\\Python\\ResearchWebscrapper\\Data Collection\\to be filtered\\2D carbon network arrange-4-1.jpeg"))
class_scores, deltaboxes = rpn(image)

# Convert the class scores and bounding box deltas to numpy arrays
class_scores = class_scores.detach().numpy()
deltaboxes = deltaboxes.detach().numpy()
print(class_scores)
# Plot the class scores
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(class_scores[ 0, :, :], cmap='viridis')
plt.title("Class Scores")

# Plot the bounding box deltas
plt.subplot(1, 2, 2)
plt.imshow(deltaboxes[ 0, :, :], cmap='viridis')
plt.title("Bounding Box Deltas")

plt.show()
