from datacollection.webscrapper import scrape
from datacollection.pullimages import pull_all_images
from cnnsorter.sorteruse import run_sorter
from rcnn.rcnnuse import rcnn_crop
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    scrape()
    pull_all_images()
    run_sorter()
    rcnn_crop()