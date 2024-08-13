from datacollection.webscrapper import scrape
from rcnn.rcnnuse import rcnn_crop
from multiprocessing import freeze_support, set_start_method
from os import path
from time import time
from shutil import rmtree
dir_path = path.dirname(path.realpath(__file__))

if __name__ == '__main__':
    #Change Linux Multithreading
    set_start_method('spawn')
    freeze_support()
    start = time()
    #Scrape articles from nature.com
    scrape()
    scrapetime = time()
    #RCNN pass to crop images based on predicted bounding boxes and keep only useful subimages
    rcnn_crop()
    rcnn = time()
    print(f"Scrape Time: {scrapetime-start}")
    print(f"Crop Time: {rcnn-scrapetime}")
    #Cleanup empty directories
    rmtree(path.join(dir_path,"datacollection","__pycache__"))
    rmtree(path.join(dir_path,"rcnn","__pycache__"))