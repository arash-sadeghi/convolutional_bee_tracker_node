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

class Rosbag2Vid:
    def __init__(self,rosbag_path,output_path,topics):
        self.bag = rosbag.Bag(rosbag_path, "r")
        self.bridge = CvBridge()
        self.output_path=output_path
        self.rosbag_path=rosbag_path
        self.topics=topics
        self.generators=[ self.bag.read_messages(_) for _ in self.topics ]
        self.images=[ 0 for _ in range(len(self.topics)) ]
    def convert(self):
        for i in range(len(self.generators)):
            try:
                topic, msg, t= next(self.generators[i])
            except Exception as E:
                print("[-] topic {} ended with exception {}".format(self.topics[i] , E ))
                continue

            if "compressed" in self.topics[i]:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg)
            #! image channels should be checked here
            cv_img=cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            self.images[i]=np.copy(cv_img)
        return t,self.get_one_image(self.images)
 
    def get_one_image(self,img_list):
            #! pure function
            padding = 10
            max_width = []
            max_height = 0
            for img in img_list:
                max_width.append(img.shape[0])
                max_height += img.shape[1]
            w = np.max(max_width)
            h = max_height + padding

            # create a new array with a size large enough to contain all the images
            final_image = np.zeros((h, w, 3), dtype=np.uint8)

            current_y = 0  # keep track of where your current image was last placed in the y coordinate
            for image in img_list:
                # add an image to the final array and increment the y coordinate
                final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
                current_y += image.shape[0]
            return final_image
    
    def close_bag(self):
        self.bag.close()

if __name__ == '__main__':
    # rosbag_path="/home/arash/catkin_ws/bagfiles/roboroyal/2022-03-29-13-53-44-teting-rotation-on-2022-03-18-22-11-04_one-hour-crop-with-homography-and-camera-info.bag"
    rosbag_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/2022-03-30-13-55-07.bag"
    topics=["/hive_0/HM_pub" , "/hive_0/crop_pub" ,  "/hive_0/kernel_pub" , "/hive_0/res_pub" ]
    output_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0]
    # output_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0] + ctime(time()).replace(":","_").replace(" ","_")
    print("[+] Extract images from {} into {} , on topics {}".format(rosbag_path, output_path , topics))
    rosbag2vid = Rosbag2Vid(rosbag_path,output_path,topics)
    #! cheat
    # start_time=1648554833.92
    # end_time=1648559505.32
    start_time=1648641307.23
    end_time=1648644831.14
    count=0
    while True:
        try:
            t , im = rosbag2vid.convert()
        except Exception as E:
            print("[-] Exception {}".format(E))
            rosbag2vid.close_bag()
            break
        t=str(t)[0:10]
        cv2.imwrite(os.path.join(output_path,str(count)+'.png'),im)
        # pdb.set_trace()
        progress= (float(t)-start_time) / (end_time-start_time) * 100
        # if int(progress)%5 == 0:
        print("[+] progress {:.2f}".format(progress),end='\r')
        count+=1
    print("\n[+] DONE!")
    
