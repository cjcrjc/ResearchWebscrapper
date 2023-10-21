import cv2
import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read bounding box data from an Excel file
bounding = True
bounding_boxes = pd.read_excel('3. RCNN/bounding_boxes.xlsx')

click_count = 0
start_x, start_y = 0, 0

# Define a mouse event handler
def Mouse_Event(event, x, y, flags, param):
    global click_count, start_x, start_y, bounding_boxes, img
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count == 0:
            start_x, start_y = x, y
            click_count += 1
        else:
            end_x, end_y = x, y
            click_count = 0
            if end_x - start_x < 0 or end_y - start_y < 0:
                print("Negative bounding box")
                return
            if abs(end_x - start_x) < 25 or abs(end_y - start_y) < 25:
                print("Small selection error")
                return
            info = {
                'image': img_path,
                'x1': start_x,
                'y1': start_y,
                'x2': end_x,
                'y2': end_y
            }
            bounding_boxes = pd.concat([bounding_boxes, pd.DataFrame([info])], ignore_index=True)
            print(f'Bounding Box added: (X1={start_x}, Y1={start_y}, X2={end_x}, Y2={end_y})')
            img = cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow('frame', img)

# Iterate over images in the "train" folder
for img_name in os.listdir(os.path.join(dir_path, "train")):
    img_path = os.path.join(dir_path, "train", img_name)
    img = cv2.imread(img_path)
    image_boxes = []
    
    # Extract bounding box information for the current image
    for i in range(len(bounding_boxes['image'])):
        if bounding_boxes['image'][i] == img_path:
            image_boxes.append([bounding_boxes['x1'][i], bounding_boxes['y1'][i], bounding_boxes['x2'][i], bounding_boxes['y2'][i]])
    
    # Draw bounding boxes on the image
    for i in range(len(image_boxes)):
        img = cv2.rectangle(img, (image_boxes[i][0], image_boxes[i][1]), (image_boxes[i][2], image_boxes[i][3]), (0, 255, 0), 2)
    
    breakout = False
    if len(image_boxes) != 0:
        pass
    else:
        # Display the image and set up the mouse event handler
        cv2.imshow('frame', img)
        cv2.setMouseCallback('frame', Mouse_Event)
        while True:
            key = cv2.waitKey(1)
            if key == 27:
                breakout = True
                break
            elif key == ord('f'):
                # Add a full bounding box covering the entire image
                info = {
                    'image': img_path,
                    'x1': 0,
                    'y1': 0,
                    'x2': img.shape[1],
                    'y2': img.shape[0]
                }
                bounding_boxes = pd.concat([bounding_boxes, pd.DataFrame([info])], ignore_index=True)
                break
            elif key in range(1, 1000):
                break
        cv2.destroyAllWindows()
        bounding_boxes.to_excel('3. RCNN/bounding_boxes.xlsx', index=False)
    
    if breakout:
        break

# Save bounding box information to an Excel file
print('Bounding box information saved to bounding_boxes.xlsx')
