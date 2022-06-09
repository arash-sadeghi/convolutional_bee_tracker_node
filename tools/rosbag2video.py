#! usr/bin/python 
import os,sys

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from time import ctime, time

import csv,datetime

class Converter:
    def __init__(self,bag_address,topic,output_dir=None):
        self.topic=topic
        self.bridge = CvBridge()
        self.output_dir=output_dir
        print("[+] Extract images from {} on topic {} to {}".format(bag_address,topic,output_dir))
        self.bag = rosbag.Bag(bag_address, "r")
        print("[+] bag successfully read")
        self.generator=self.bag.read_messages(self.topic)
    
    def convert(self):
        total_msg_num=self.bag.get_message_count(self.topic)
        count=0
        for topic, msg, t in self.generator:
            time_tmp=t.to_sec()
            if "compressed" in self.topic:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg)
            
            if len(cv_img.shape)<3:
                cv_img=cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            if not self.output_dir is None:
                cv2.imwrite( os.path.join(self.output_dir,'{:06d}.png'.format(count)) , cv_img)
            count += 1
            percentage=count/(total_msg_num)*100
            print("[+] progress {:.2f}".format(percentage),end='\r')

        self.bag.close()
        print("[+] progress {} / {}= {:.2f}".format(count, total_msg_num,percentage))


        
if __name__ == '__main__':
    bag_address="/home/arash/catkin_ws/bagfiles/roboroyal/2022-03-18-22-11-04_one-hour-crop-with-homography-and-camera-info.bag"
    topic="/hive_0/cropped_image/compressed"
    output_dir="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir,exist_ok=True)    
    conv=Converter(bag_address,topic,output_dir)
    conv.convert()

















