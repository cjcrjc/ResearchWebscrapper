import torch, glob, time, pathlib, os, shutil
from torch.autograd import Variable
from PIL import Image
import numpy as np
from multiprocessing import Process, cpu_count, freeze_support
import cnnsorter.sortercnn as sorter
dir_path = os.path.dirname(os.path.realpath(__file__))

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

def sort(images_path,classes,model, filtered_save_dir):
    i=0
    for image in images_path:
        i+=1
        if prediction(image, sorter.transformer,model,classes) == classes[0]:
            print("[+] Passed:", os.path.basename(image))
            shutil.move(image, filtered_save_dir)
        else:
            print("[-] Failed:", os.path.basename(image))
            os.remove(image)
   
def run_sorter():
    freeze_support()
    cores = cpu_count()
    images_path=glob.glob(os.getcwd()+'/datacollection/images/*')
    images_paths = np.array_split(images_path,cores)
    filtered_save_dir = os.path.join(dir_path,'filtered')
    if not os.path.isdir(filtered_save_dir):
        os.mkdir(filtered_save_dir)
        
    #Unzip Data
    data_path = os.path.join(dir_path,"data")
    shutil.unpack_archive(data_path+".zip", data_path)   
    #Categories
    classes=sorted([j.name.split('/')[-1] for j in pathlib.Path(data_path+"/test").iterdir()])
    print(classes)

    #Load Model
    checkpoint=torch.load(os.path.join(dir_path,'best.model'))
    model=sorter.SorterCNN(num_classes=2)
    model.load_state_dict(checkpoint)
    model.eval()

    processes = []
    for i in range(cores):
        p = Process(target=sort, args=(images_paths[i],classes,model, filtered_save_dir))
        p.start()
        processes.append(p)
        time.sleep(2)

    for process in processes:
        process.join()

    #Zip data
    shutil.make_archive(data_path, 'zip', data_path)
    shutil.rmtree(data_path)

