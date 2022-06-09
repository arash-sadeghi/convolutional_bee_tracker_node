#! usr/bin/python 
import os,sys
import pdb
import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from time import ctime, time

import csv,datetime

class Rosbag2txt:
    def __init__(self,rosbag_path,output_path,topics):
        self.bag = rosbag.Bag(rosbag_path, "r")
        self.output_path=output_path
        self.rosbag_path=rosbag_path
        self.topics=topics
        self.generators= [self.bag.read_messages(_) for _ in self.topics]
    def convert(self):
        try:
            #! i=0 cheat
            topic, msg, t= next(self.generators[0])
        except Exception as E:
            print("[-] topic {} ended with exception {}".format(self.topics , E ))
            return None,None 
        # pdb.set_trace()
        return t,float(msg.data)
 
    
    def close_bag(self):
        self.bag.close()
def standardize(inp):
    if inp > 360:
        inp=inp - 360
    elif inp < 0:
        inp = inp +360
    inp=round(inp,2)
    return inp

if __name__ == '__main__':
    rosbag_path="/home/arash/catkin_ws/bagfiles/roboroyal/2022-03-30-13-55-07-recorded-results.bag"
    topics=["/hive_0/rot_pub"]
    output_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0]
    # output_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0] + ctime(time()).replace(":","_").replace(" ","_")
    print("[+] Extract rotations from {} into {} , on topics {}".format(rosbag_path, output_path , topics))
    rosbag2txt = Rosbag2txt(rosbag_path,output_path,topics)
    start_time=1648641307.23
    end_time=1648644831.14
    count=0
    #! cheat
    initial_rot=314

    all_rots=[]
    while True:
        try:
            t , rot = rosbag2txt.convert()
        except Exception as E:
            print("[-] Exception {}".format(E))
            rosbag2txt.close_bag()
            break
        if not t is None:
            t=str(t)[0:10]
            
            #! cheat
            initial_rot+=(-rot)

            initial_rot=standardize(initial_rot)
            all_rots.append(initial_rot)
            print(count,initial_rot)
            progress= (float(t)-start_time) / (end_time-start_time) * 100
            print("[+] progress {:.2f}".format(progress),end='\r')
            count+=1
        else: #! end of bag
            #! cheat of topics[0]
            np.savetxt(os.path.join(output_path , topics[0].replace("/","_")[1:]+".txt"),np.array(all_rots),fmt='%f')
            break
    print("\n[+] DONE!")
    
