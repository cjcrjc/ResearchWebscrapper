import os
import shutil
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Create folders for sorting
categories = ['Keep', 'Not']

# Function to display and sort images
def sort_images(image_paths, current_index):
    if current_index < len(image_paths):
        image_path = image_paths[current_index]

        # Open and display the image using tkinter
        root = tk.Tk()
        img = Image.open(image_path)
        img.show()
        #root.destroy()

        # Get keyboard input for sorting
        print(f"Press a key to sort '{image_path}' into a category:")
        print("1: Keep, 2: Not, Q: Quit")
        choice = input().strip().lower()

        # Sort the image into the corresponding category
        if choice == '1':
            shutil.move(image_path, os.path.join('Keep', os.path.basename(image_path)))
        elif choice == '2':
            shutil.move(image_path, os.path.join('Not', os.path.basename(image_path)))
        elif choice == 'q':
            exit()  # Quit the program if 'q' is pressed
        else:
            print("Invalid choice. Image not sorted.")

        # Continue with the next image
        sort_images(image_paths, current_index + 1)
    else:
        print("All images processed.")

# Ask the user to select a directory containing images
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
directory_path = filedialog.askdirectory(title="Select the directory containing images")

if not directory_path:
    print("No directory selected. Exiting.")
    exit()

# Get a list of all image files in the selected directory
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
image_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if
               os.path.isfile(os.path.join(directory_path, file)) and
               os.path.splitext(file)[-1].lower() in image_extensions]

# Start sorting images
sort_images(image_paths, 0)