import shutil
from os import path, remove

def zip(path):
    shutil.make_archive(path, 'zip', path)
    remove(path)
