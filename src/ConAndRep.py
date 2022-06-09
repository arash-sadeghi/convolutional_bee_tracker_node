#! /usr/bin/env python
import sys #! not added to dependencies
import pdb #! not added to dependencies
import numpy as np #! not added to dependencies
import cv2 #! not added to dependencies
# from cv2 import imwrite, equalizeHist , matchTemplate , TM_CCORR_NORMED , COLOR_BGR2GRAY , cvtColor #! not added to dependencies
import matplotlib.pyplot as plt #! not added to dependencies
from scipy.ndimage.interpolation import rotate #! not added to dependencies
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image , CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from rr_msgs.msg import BeePosition , BeePositionArray
# from torch.nn import functional as F
# import torch 
# def manual_conv2d(inp,ker):
#     inp=torch.tensor(im_cropped)
#     inp=inp.reshape((1,1,inp.shape[0],inp.shape[1]))
#     ker=torch.tensor(queen-np.mean(queen[:]))
#     ker=ker.reshape((1,1,ker.shape[0],ker.shape[1]))
#     out=F.conv2d(inp,ker)
#     return out[0,0,:,:].numpy()

QUEUE_SIZE=1
# CROP_WIDTH=400
#! cheat 300
CROP_WIDTH=300

def put_in_canvas(queen,big_dim,center):
    fixed_queen_dim=queen.shape
    canvas=np.zeros((big_dim[0],big_dim[1]))
    queen_center=[0,0]
    if center:
        queen_center[0]=(big_dim[0]-fixed_queen_dim[0])//2
        queen_center[1]=(big_dim[1]-fixed_queen_dim[1])//2
    else:
        queen_center[0]=(big_dim[0]-fixed_queen_dim[0])
        queen_center[1]=(big_dim[1]-fixed_queen_dim[1])//2

    canvas[ queen_center[0]:queen_center[0]+fixed_queen_dim[0] , queen_center[1]:queen_center[1]+fixed_queen_dim[1] ] = queen
    queen = np.copy(canvas)
    return queen

def ROT(inp,a):
    return rotate(inp,a,mode='nearest')

def generate_rotations(queen,rotations,big_dim,fixed_queen_dim,queen_filt):
    out=np.zeros((len(rotations),big_dim[0],big_dim[1]))
    masks=np.zeros((len(rotations),big_dim[0],big_dim[1]))
    for c,v in enumerate(rotations):
        rotated = ROT(queen, v)
        rotated = rotated [ (rotated.shape[0]-big_dim[0])//2 : (rotated.shape[0]-big_dim[0])//2+big_dim[0] ,
         (rotated.shape[1]-big_dim[1])//2 : (rotated.shape[1]-big_dim[1])//2+big_dim[1]]

        rotated_mask=ROT(queen_filt,v)

        rotated_mask = rotated_mask [ (rotated_mask.shape[0]-big_dim[0])//2 : (rotated_mask.shape[0]-big_dim[0])//2+big_dim[0] ,
         (rotated_mask.shape[1]-big_dim[1])//2 : (rotated_mask.shape[1]-big_dim[1])//2+big_dim[1]]

        out[c , 0:rotated.shape[0] , 0:rotated.shape[1] ] = rotated
        masks[c , 0:rotated.shape[0] , 0:rotated.shape[1] ] = rotated_mask

    return out , masks

def normalize(inp):
    return (inp-np.min(inp[:])) / (np.max(inp[:])-np.min(inp[:]))*255


def crop_queen(im,queen,queen_pos,queen_filt,rot_track,rotations,CROP_WIDTH,ros_agent,big_dim,fixed_queen_dim):
    '''
    im : the whole image
    queen : kernel to be convoluted with image
    queen_pos : position of the corner of the queen kernel in im
    queen_filt : initial mask to create rotated masks
    rot_track : absolute rotation of queen
    rotations : possible relative rotations
    CROP_WIDTH : width of the neighbourhood for searching kernel
    '''
    #! putting class vars in local vars
    im=im.astype(np.float32)
    queen=queen.astype(np.float32)

    #! taking a crop around queen 
    crop_points=np.zeros((2,2))
    crop_points[0,0]=max( im.shape[0]//2 - CROP_WIDTH//2,0)
    crop_points[0,1]=min( crop_points[0,0]+CROP_WIDTH,im.shape[0])
    crop_points[1,0]=max( im.shape[1]//2  - CROP_WIDTH//2,0)
    crop_points[1,1]=min( crop_points[1,0]+CROP_WIDTH,im.shape[1])
    crop_points=crop_points.astype(np.uint32)
    im_cropped=im[crop_points[0,0]:crop_points[0,1],crop_points[1,0]:crop_points[1,1]]
    ros_agent.crop_publish(im_cropped)
    #! rotating queen in possible rotations
    queen_stack , masks=generate_rotations(queen,rotations+rot_track,big_dim,fixed_queen_dim,queen_filt)
    #! convolving all possible positions of queen through cropped image
    match_stack=[]
    maxes=[]
    KERNELS=[]
    for i in range(queen_stack.shape[0]):
        filt=np.round(masks[i])
        kernel=np.copy(queen_stack[i])
        kernel[filt==1]=kernel[filt==1]-np.mean(kernel[filt==1])
        kernel=kernel.astype(np.float32)
        KERNELS.append(kernel)
        HM = cv2.matchTemplate(im_cropped,kernel,cv2.TM_CCORR_NORMED) #! 1 best
        # HM = manual_conv2d(im_cropped,kernel)
        match_stack.append(HM)
        maxes.append(match_stack[i].max())
    maxes=np.array(maxes)
    chosen_rot=maxes.argmax()
    matched = match_stack[ chosen_rot ]
    matched=normalize(matched)
    ros_agent.HM_publish(matched)

    rot_track+=rotations[chosen_rot]
    # ros_agent.rot_publish(str( rotations[chosen_rot] ))
    ros_agent.rot_publish(str( rot_track ))
    # ros_agent.added_rot_msg_publish(rot_track)
    #! getting the position of queen based on heatmap
    indx=np.unravel_index(matched.argmax(),matched.shape)
    #! updating queen position also will track position change of queen.
    #! which is not needed in our case since queen is always in the middle
    # queen_pos[0,0]= min( crop_points[0,0] + indx[0] , im.shape[0])
    # queen_pos[0,1]= min( queen_pos[0,0] + queen_stack.shape[1] , im.shape[0])
    # queen_pos[1,0]= min(crop_points[1,0] + indx[1] , im.shape[1])
    # queen_pos[1,1]= min(queen_pos[1,0] + queen_stack.shape[2] , im.shape[1])

    res=im_cropped[indx[0]:indx[0]+queen_stack.shape[1] , indx[1]:indx[1]+queen_stack.shape[2]]

    res[np.round(masks[chosen_rot])<=0]=0
    # queen=cv2.equalizeHist(queen)
    res=res.astype(np.float32)
    ros_agent.kernel_publish(KERNELS[chosen_rot])
    return res,rot_track


class Ros_Agent:
    def __init__(self,sub_topic):
        self.sub_topic=sub_topic
        if "compressed" in self.sub_topic["im"]:
            self.im_sub=rospy.Subscriber(sub_topic["im"], CompressedImage, self.callback_in_im,queue_size=21680)
        else:
            self.im_sub=rospy.Subscriber(sub_topic["im"], Image, self.callback_in_im)
        if "pos" in self.sub_topic.keys():
            #! do this if beepos is supposed to be published
            self.pos_sub=rospy.Subscriber(sub_topic["pos"], BeePositionArray, self.callback_queen_position)

        
        self.bridge = CvBridge()
        self.im=0
        self.first_im=True

        #! for rospy default queue size is 10
        self.res_pub = rospy.Publisher("res_pub", Image, queue_size=QUEUE_SIZE)
        self.kernel_pub = rospy.Publisher("kernel_pub", Image, queue_size=QUEUE_SIZE)
        self.crop_pub = rospy.Publisher("crop_pub", Image, queue_size=QUEUE_SIZE)
        self.HM_pub=rospy.Publisher("HM_pub", Image, queue_size=QUEUE_SIZE)
        self.rot_pub=rospy.Publisher("rot_pub", String, queue_size=QUEUE_SIZE)
        self.added_rot_msg=rospy.Publisher("queen_position_angle", BeePositionArray, queue_size=QUEUE_SIZE)

        self.rotations=np.arange(-6,6+1,1)
        node_name=sys.argv[0].split("/")[-1].split(".")[0]
        rospy.init_node(node_name, anonymous=True)
        self.queen=np.load(sys.argv[1]+sys.argv[2])
        self.fixed_queen_dim=self.queen.shape
        self.big_dim=[0,0]
        # self.big_dim[0]=int(np.sqrt(self.queen.shape[0]**2+self.queen.shape[1]**2))+1
        # self.big_dim[1]=int(np.sqrt(self.queen.shape[0]**2+self.queen.shape[1]**2))+1
        #! cheat 150
        self.big_dim[0]=150*2
        self.big_dim[1]=150*2

        self.queen_square=put_in_canvas(self.queen,self.big_dim,center=False)
    def callback_queen_position(self,data):
        if len(data.positions) > 0 :
            data.positions[0].alpha=self.rot_track
            self.added_rot_msg.publish(data)

    def callback_in_im(self,data):
        if "compressed" in self.sub_topic["im"]:
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            except CvBridgeError as E:
                print("[-] Error Occured {}".format(E))
        else:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data)
            except CvBridgeError as E:
                print("[-] Error Occured {}".format(E))
        
        #! convert image only if it has three chaneels
        if len(cv_image.shape)>2:
            cv_image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY) 
            print("[-] receiving 3D image")
        self.im=np.copy(cv_image)

        if self.first_im: #! this is first frame so initilized kernel
            self.first_im=False
            init_pos_crop=self.im[ (self.im.shape[0]-CROP_WIDTH)//2:(self.im.shape[0]+CROP_WIDTH)//2 , (self.im.shape[1]-CROP_WIDTH)//2:(self.im.shape[1]+CROP_WIDTH)//2 ]
            self.queen , self.queen_pos , self.rot_track , self.queen_filt =get_first_queen( init_pos_crop ,self.queen_square , self.big_dim , self.fixed_queen_dim)
            self.queen_pos[0,:]=self.queen_pos[0,:]+(self.im.shape[0]-CROP_WIDTH)//2
            self.queen_pos[1,:]=self.queen_pos[1,:]+(self.im.shape[1]-CROP_WIDTH)//2
        
        else: #! this is not first frame so proceed with tracking
            res,self.rot_track = crop_queen(self.im,self.queen,self.queen_pos,self.queen_filt,self.rot_track,self.rotations,CROP_WIDTH,self,self.big_dim,self.fixed_queen_dim)
            self.result_publish(res)
        return self.im

    def result_publish(self,data):
        data_cv2=self.bridge.cv2_to_imgmsg(data)
        self.res_pub.publish(data_cv2)

    def kernel_publish(self,data):
        data_cv2=self.bridge.cv2_to_imgmsg(data)
        self.kernel_pub.publish(data_cv2)

    def crop_publish(self,data):
        data_cv2=self.bridge.cv2_to_imgmsg(data)
        self.crop_pub.publish(data_cv2)

    def HM_publish(self,data):
        data_cv2=self.bridge.cv2_to_imgmsg(data)
        self.HM_pub.publish(data_cv2)
    def rot_publish(self,data):
        self.rot_pub.publish(data)

def get_first_queen(im,kernel , big_dim , fixed_queen_dim):
    #! manual
    # queen_pos=np.array( [[ 514 , 711  ] , [ 457 , 654 ]])
    # queen=np.load("queen.npy")
    # rot_track=-np.load("angle.npy")
    
    #! automatic
    initial_rotations=np.arange(0,360,10)
    queen_pos=np.zeros((2,2))
    maxes=[]
    HMS=[]
    im=im.astype(np.float32)

    white=np.ones(fixed_queen_dim)
    mask=put_in_canvas(white,big_dim,center=False)

    queen_stack , masks=generate_rotations(kernel,initial_rotations,big_dim,fixed_queen_dim,mask)

    for c in range(queen_stack.shape[0]):
        filt=np.round(masks[c])
        kernel=np.copy(queen_stack[c])
        kernel[filt==1]=kernel[filt==1]-np.mean(kernel[filt==1])
        kernel=kernel.astype(np.float32)
        HM = cv2.matchTemplate(im,kernel,cv2.TM_CCORR_NORMED) #! 1 best
        HMS.append(HM)
        maxes.append(HM.max())
    maxes=np.array(maxes)
    chosen_rot=maxes.argmax()
    rot_track=initial_rotations[chosen_rot]
    pos=np.unravel_index(HMS[chosen_rot].argmax() , HMS[chosen_rot].shape)
    queen_pos[0,0]=pos[0]
    queen_pos[0,1]=pos[0]+big_dim[0]
    queen_pos[1,0]=pos[1]
    queen_pos[1,1]=pos[1]+big_dim[1]

    queen_pos=queen_pos.astype(np.uint32)
    queen=queen_stack[chosen_rot]
    return queen , queen_pos , rot_track , masks[chosen_rot]

if __name__=='__main__':
    print("[+] passed arguments ",sys.argv)
    # sub_topic={"im" : sys.argv[3][sys.argv[3].index("=")+1:]
    # , "pos" : sys.argv[4][sys.argv[4].index("=")+1:]}

    sub_topic={"im" : sys.argv[3][sys.argv[3].index("=")+1:]}
    ros_agent=Ros_Agent(sub_topic)



    rospy.spin()


# # 1647349624.842732
# # 1647349725.592460
""" mem usage
without queen_tracker.crop_queen -> 225M
with only libraries -> 133M
with all 300M
 """
