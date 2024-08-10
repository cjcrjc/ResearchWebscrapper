from datacollection.webscrapper import scrape
from cnnsorter.sorteruse import run_sorter
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
    #Primilinary binary sort of images to keep useful ones
    run_sorter()
    cnnsorter = time()
    #RCNN pass to crop images based on predicted bounding boxes and keep only useful subimages
    rcnn_crop()
    rcnn = time()
    print(f"Scrape Time: {scrapetime-start}")
    print(f"Sort Time: {cnnsorter-scrapetime}")
    print(f"Crop Time: {rcnn-cnnsorter}")
    #Cleanup empty directories
    rmtree(path.join(dir_path,"datacollection","images"))
    rmtree(path.join(dir_path,"datacollection","__pycache__"))
    rmtree(path.join(dir_path,"cnnsorter","filtered"))
    rmtree(path.join(dir_path,"cnnsorter","__pycache__"))
    rmtree(path.join(dir_path,"rcnn","__pycache__"))