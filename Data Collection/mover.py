import os, shutil, glob, random

dir_path = os.path.dirname(os.path.realpath(__file__))

keeps = os.path.join(dir_path, "keep")
nots = os.path.join(dir_path, "not")

keeps = glob.glob(keeps + "/*.*")
nots = glob.glob(nots + "/*.*")
# print(keeps)
# print(nots)

for img in keeps:
    if random.random() < .85:
        try:
            shutil.move(img, os.path.join(dir_path, "data/train/Nanostructure"))
        except Exception as e:
            os.remove(img)
    else:
        try:
            shutil.move(img, os.path.join(dir_path, "data/test/Nanostructure"))
        except Exception as e:
            os.remove(img)
        
# for img in nots:
#     if random.random() < .85:
#         try:
#             shutil.move(img, os.path.join(dir_path, "data/test/Not"))
#         except Exception as e:
#             os.remove(img)
#     else:
#         try:
#             shutil.move(img, os.path.join(dir_path, "data/test/Nanostructure"))
#         except Exception as e:
#             os.remove(img)
        