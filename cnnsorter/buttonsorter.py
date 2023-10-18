import os
import shutil
import cv2
import tkinter as tk
from tkinter import filedialog
from pynput.keyboard import *

dir_path = os.path.dirname(os.path.realpath(__file__))

# Global variable to keep track of the current index
current_index = 0

# Function to sort images
def sort_images(image_paths):
    global current_index  # Use the global current_index variable
    while current_index < len(image_paths):
        image_path = image_paths[current_index]
        img = cv2.imread(image_path)
        cv2.imshow(f"Image {current_index + 1}", img)
        key = cv2.waitKey(0)  # Wait for a key press (in milliseconds)
        cv2.destroyAllWindows()
        if key == ord('f'):
            shutil.move(image_path, os.path.join(dir_path, 'keep', os.path.basename(image_path)))
        elif key == ord('g'):
            shutil.move(image_path, os.path.join(dir_path, 'not', os.path.basename(image_path)))
        elif key == ord('q'):
            exit()  # Quit the program if 'q' is pressed
        else:
            print("Invalid choice. Image not sorted.")
        current_index += 1

# Ask the user to select a directory containing images
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
directory_path = filedialog.askdirectory(title="Select the directory containing images")
root.destroy()
if not directory_path:
    print("No directory selected. Exiting.")
    exit()
    
# Get a list of all image files in the selected directory
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
image_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if
               os.path.isfile(os.path.join(directory_path, file)) and
               os.path.splitext(file)[-1].lower() in image_extensions]

# Start sorting images
sort_images(image_paths)

print("All images sorted and moved.")
