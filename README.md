# ResearchWebscrapper
#### Summary
This repository contains a collection of Python scripts and programs for various image processing and deep learning sorters. The code provided covers the following functionalities:
-Data collection from webscrappers
-Image extraction from scientific articles
-A trained binary classification CNN to remove unwanted images
-A trained RCNN to crop desired images to just desired subimages

#### Usage
To use these scripts, make sure to install the required dependancies with:
pip install -r requirements.txt

You also need to have ChromeWebdriver installed in PATH environment variables
Then just run the run.py file in the main directory and provide the desired search keyword when prompted. It will take a few minutes to process end-to-end but after it's finished there will be a folder, rcnn/cropped, containing desired SEM images.

#### Functionality
Web Scraping (webscrapper.py):
- Utilizes BeautifulSoup and Selenium libraries to scrape scientific articles from nature.com.
- Downloads PDF files of the articles.

Image Extraction (pullimages.py):
- Extracts images from the downloaded PDF files using PyMuPDF library (not explicitly shown in the code but implied by the function name and context).

Binary Image Sorting (sorteruse.py):
- Uses a binary classification model (presumably a neural network model) to sort images into two categories: 'nanostructure' and 'not'.
- Moves images classified as 'nanostructure' to a 'filtered' directory for further processing.

RCNN Image Cropping (rcnnuse.py):
- Implements a Faster R-CNN (Region-based Convolutional Neural Network) model to predict bounding boxes around objects in images.
- Uses these predicted bounding boxes to crop the images, keeping only the relevant subimages.