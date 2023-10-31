# ResearchWebscrapper
Summary
This repository contains a collection of Python scripts and programs for various image processing and deep learning sorters. The code provided covers the following functionalities:
-Data collection from webscrappers
-Image extraction from scientific articles
-A trained binary classification CNN to remove unwanted images
-A trained RCNN to crop desired images to just desired subimages

Usage
To use these scripts, make sure to install the required dependancies with:
pip install torch torchvision opencv-python pynput matplotlib numpy Pillow multiprocess pymupdf requires.io PySimpleGUI selenium bs4 requests pandas scikit-learn frontend
sudo apt-get install python3-tk
Then just run the run.py file in the main directory and provide the desired search keyword when prompted. It will take a few minutes to process end-to-end but after it's finished there will be a folder, rcnn/cropped, containing desired SEM images.