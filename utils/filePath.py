import os
import shutil

def createPath(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def deletePath(filepath, remain=True):
    if remain:
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        shutil.rmtree(filepath)