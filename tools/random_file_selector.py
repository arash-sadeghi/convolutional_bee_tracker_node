import os
import numpy as np
import random
import shutil

def random_select(in_path,out_path,choice_num):
    files=os.listdir(in_path)
    selected = random.choices( files , k=choice_num)
    for c,f in enumerate(selected):
        source = os.path.join(in_path,f)
        target =  os.path.join(out_path,f)
        shutil.copyfile( source , target)
        print("[+] progress {:.2f}".format( c/len(selected) * 100 ))

if __name__=='__main__':
    im_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/rosbag2video"
    out_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/DenseObjectAnnotation/static/png"
    random_select(im_path , out_path , 200)