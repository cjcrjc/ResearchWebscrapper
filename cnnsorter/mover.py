import os, shutil, glob, random

# Get the path of the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define the source directories for 'keep' and 'not' images
keeps = os.path.join(dir_path, "keep")
nots = os.path.join(dir_path, "not")

# Get a list of file paths in the 'keep' and 'not' directories
keeps = glob.glob(keeps + "/*.*")
nots = glob.glob(nots + "/*.*")

# Iterate through 'keep' images
for img in keeps:
    if random.random() < 0.85:  # 85% chance of moving to train/nanostructure
        try:
            # Move the image to the 'train/nanostructure' directory
            shutil.move(img, os.path.join(dir_path, "data/train/nanostructure"))
        except Exception as e:
            # If there's an exception (e.g., file already exists), remove the image
            os.remove(img)
    else:
        try:
            # Move the image to the 'test/nanostructure' directory
            shutil.move(img, os.path.join(dir_path, "data/test/nanostructure"))
        except Exception as e:
            os.remove(img)

# Iterate through 'not' images
for img in nots:
    if random.random() < 0.85:  # 85% chance of moving to train/not
        try:
            # Move the image to the 'train/not' directory
            shutil.move(img, os.path.join(dir_path, "data/train/not"))
        except Exception as e:
            os.remove(img)
    else:
        try:
            # Move the image to the 'test/not' directory
            shutil.move(img, os.path.join(dir_path, "data/test/not"))
        except Exception as e:
            os.remove(img)
