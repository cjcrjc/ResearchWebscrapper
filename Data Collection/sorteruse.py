import torch
from torch.autograd import Variable
from PIL import Image
import pathlib
import glob
from sortercnn import *
import numpy as np
from multiprocessing import Process, set_start_method
import os, shutil

#prediction function
def prediction(img_path,transformer,model,classes):
    image=Image.open(img_path)
    image_tensor=transformer(image).float()
    image_tensor=image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()
    input=Variable(image_tensor)
    output=model(input)
    index=output.data.numpy().argmax()
    pred=classes[index]
    return pred

def sort(images_path,classes,model):
    i=0
    for image in images_path:
        i+=1
        if prediction(image, transformer,model,classes) == classes[0]:
            shutil.move(image, os.path.join(dir_path, "keep"))
        else:
            shutil.move(image, os.path.join(dir_path, "not"))

    
if __name__ == '__main__':
    set_start_method('spawn')
    cores = 6
    #Categories
    train_path=dir_path + '/data/train'
    test_path=dir_path + '/data/test'
    images_path=glob.glob(dir_path+'/to be filtered'+'/*')
    images_paths = np.array_split(images_path,cores)
    filtered_save_dir = os.path.join(dir_path,'filtered')
    if not os.path.isdir(filtered_save_dir):
        os.mkdir(filtered_save_dir)

    root=pathlib.Path(train_path)
    classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
    print(classes)

    #Load Model
    checkpoint=torch.load('best.model')
    model=SorterCNN(num_classes=2)
    model.load_state_dict(checkpoint)
    model.eval()
    
    #sort(images_path,classes)

    processes = []
    for i in range(cores):
        p = Process(target=sort, args=(images_paths[i],classes,model))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()