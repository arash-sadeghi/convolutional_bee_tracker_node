#! /usr/bin/env python3
import os
import sys
import pdb
import numpy as np 
import cv2 
# from cv2 import imwrite, equalizeHist , matchTemplate , TM_CCORR_NORMED , COLOR_BGR2GRAY , cvtColor 
import matplotlib.pyplot as plt 
from scipy.ndimage.interpolation import rotate 
from time import ctime , time
#! modules for ros
# import rospy
# from std_msgs.msg import String
# from sensor_msgs.msg import Image , CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# from rr_msgs.msg import BeePosition , BeePositionArray

from torch.nn import functional as F
import torch 
torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("[+] pytorch device is {}".format(torch_device))

def manual_conv2d(inp,ker):
    #! sending tensor to cuda fastens code with 0.06 s
    inp=torch.tensor(inp)
    # inp=torch.tensor(inp).to(torch_device)
    # ker=torch.tensor(ker).to(torch_device)
    # out=F.conv2d(ker,inp)
    inp=inp.reshape((1,1,1,inp.shape[0],inp.shape[1]))
    out=F.conv3d(ker , inp.float())
    out=out.squeeze()
    #! sending tensor to cpu increases time by 0.06
    # out=out.cpu()
    # out = out.numpy()

    return out

CODE_START_TIME = ctime(time()).replace(":","_").replace(" ","_")
QUEUE_SIZE=1
# CROP_WIDTH=400
#! cheat 300
CROP_WIDTH=300
MARKER_CROP_WIDTH=300
MARKER_DISPALCEMENT_WARNING_THRESH=100
# DEVICE="ros"
DEVICE="thinkpad"
# DEVICE="server"

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

def crop_queen(im,queen,rot_track,rotations,CROP_WIDTH,queen_stack , masks , drift=None):
    '''
    im : the whole image
    queen : kernel to be convoluted with image
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
    if drift is None: #! assumed that there is no drift
        crop_points[0,0]=max( im.shape[0]//2 - CROP_WIDTH//2,0)
        crop_points[0,1]=min( crop_points[0,0]+CROP_WIDTH,im.shape[0])
        crop_points[1,0]=max( im.shape[1]//2  - CROP_WIDTH//2,0)
        crop_points[1,1]=min( crop_points[1,0]+CROP_WIDTH,im.shape[1])
    else : #! drift taken into account
        crop_points[0,0]=max( im.shape[0]//2 - CROP_WIDTH//2 + drift[0] , 0)
        crop_points[0,1]=min( crop_points[0,0]+CROP_WIDTH , im.shape[0])
        crop_points[1,0]=max( im.shape[1]//2  - CROP_WIDTH//2 + drift[1], 0)
        crop_points[1,1]=min( crop_points[1,0]+CROP_WIDTH,im.shape[1])

    crop_points=crop_points.astype(np.uint32)
    im_cropped=im[crop_points[0,0]:crop_points[0,1],crop_points[1,0]:crop_points[1,1]]
    #! convolving all possible positions of queen through cropped image
    match_stack=[]
    maxes=[]
    KERNELS=[]
    
    #! 3d Convolution test
    kernel = queen_stack 
    filt=np.copy( np.round(masks) )
    """
    #! this loops takes 0.3 s DONT DELTE
    for i in range(filt.shape[0]):
        kernel[i , filt[i]==1 ] = kernel[i , filt[i]==1 ] - np.average( kernel[i , filt[i]==1 ] )
    """
    HM2=manual_conv2d(im_cropped , kernel)
    chosen_rot=HM2.argmax()

    """
    #! faster convolution
    tmp=np.zeros((queen_stack.shape[1] , queen_stack.shape[0]*queen_stack.shape[2]))
    i=0
    while i<tmp.shape[1]:
        array_index=i//queen_stack.shape[1]
        filt=np.round(masks[ array_index ])
        kernel = np.copy(queen_stack[ array_index ])
        kernel[ filt==1 ] = kernel[ filt==1 ] - np.average(kernel[ filt==1 ])
        tmp[: , i:i+queen_stack.shape[1] ] = np.copy(kernel)
        i=i+queen_stack.shape[1]
    tmp=tmp.astype(np.float32)
    t1=time()
    HM2 = cv2.matchTemplate(im_cropped , tmp , cv2.TM_CCORR_NORMED).squeeze()

    print('conv time ',time()-t1)

    indexes=np.arange(0,tmp.shape[1] ,tmp.shape[0])
    HM2_selected = HM2[indexes]
    chosen_rot=HM2_selected.argmax()
    matched = tmp[ : , chosen_rot*queen_stack.shape[1]: chosen_rot*queen_stack.shape[1]*2 ]
    """

    #! old convolution method
    """
    for i in range(queen_stack.shape[0]):
        filt=np.round(masks[i])
        kernel=np.copy(queen_stack[i])
        kernel[filt==1]=kernel[filt==1]-np.mean(kernel[filt==1])
        kernel=kernel.astype(np.float32)
        KERNELS.append(kernel)
        t1=time()
        HM = cv2.matchTemplate(im_cropped,kernel,cv2.TM_CCORR_NORMED) #! 1 best
        print('conv time ',time()-t1)
        # HM = manual_conv2d(im_cropped,kernel)
        match_stack.append(HM)
        maxes.append(match_stack[i].max())
    maxes=np.array(maxes)
    chosen_rot=maxes.argmax()
    match_stack=np.array(match_stack).squeeze()
    matched = match_stack[ chosen_rot ]
    """ 

    #! for relative rots:
    # rot_track+=rotations[chosen_rot]

    #! for absolute rots:
    rot_track = np.copy(rotations[chosen_rot])

    
    #! old way when there was also displacement
    #! getting the position of queen based on heatmap
    # indx=np.unravel_index(matched.argmax(),matched.shape)
    #! updating queen position also will track position change of queen.
    #! which is not needed in our case since queen is always in the middle
    # res=im_cropped[indx[0]:indx[0]+queen_stack.shape[1] , indx[1]:indx[1]+queen_stack.shape[2]]
    # res[np.round(masks[chosen_rot])<=0]=0
    #! new way with concentration on rotation
    res = np.copy(im_cropped)
    res[np.round(masks[chosen_rot])<=0]=0
    
    res=res.astype(np.float32)
    return res,rot_track

def get_first_queen(im,kernel , big_dim , fixed_queen_dim ,rotations , masks , queen_stack):
    maxes=[]
    HMS=[]
    im=im.astype(np.float32)

    white=np.ones(fixed_queen_dim)
    mask=put_in_canvas(white,big_dim,center=False)
    if (masks is None) or (queen_stack is None):
        queen_stack , masks=generate_rotations(kernel,rotations,big_dim,fixed_queen_dim,mask)

    for c in range(queen_stack.shape[0]):
        kernel=np.copy(queen_stack[c])
        """ uploaded kernel is already substracted from average no need for this
        filt=np.round(masks[c])
        kernel[filt==1]=kernel[filt==1]-np.mean(kernel[filt==1])
        """
        kernel=kernel.astype(np.float32)
        HM = cv2.matchTemplate(im,kernel,cv2.TM_CCORR_NORMED) #! 1 best
        HMS.append(HM)
        maxes.append(HM.max())
    maxes=np.array(maxes)
    chosen_rot=maxes.argmax()
    rot_track=rotations[chosen_rot]
    queen=queen_stack[chosen_rot]
    return queen , rot_track , masks[chosen_rot] ,queen_stack , masks

def find_marker(im,marker):
    #! this function return position of center of marker in 300*300 crop of the middle
    marker_position=None

    im_cropped=im[ (im.shape[0]-MARKER_CROP_WIDTH)//2:(im.shape[0]+MARKER_CROP_WIDTH)//2 , (im.shape[1]-MARKER_CROP_WIDTH)//2:(im.shape[1]+MARKER_CROP_WIDTH)//2 ]
    im_cropped=im_cropped.astype(np.float32)
    marker=marker.astype(np.float32)

    marker=marker-np.average(marker)
    HM = cv2.matchTemplate(im_cropped,marker,cv2.TM_CCORR_NORMED) #! 1 best
    marker_position = np.array(np.where(HM==HM.max())).squeeze()
    marker_position = marker_position + np.array(marker.shape) //2
    marker_position_drift = marker_position - np.array([ MARKER_CROP_WIDTH//2 , MARKER_CROP_WIDTH//2 ])

    if abs(marker_position_drift[0]) > MARKER_DISPALCEMENT_WARNING_THRESH or abs(marker_position_drift[1]) > MARKER_DISPALCEMENT_WARNING_THRESH:
        # print("what")
        pass

    return np.copy(marker_position_drift)

class Convolve_Agent:
    def __init__(self,im_path="." , out_path="." , kernel_path="." , queen_stack_path = None , masks_path = None , marker_path=None):
        self.out_path=out_path
        self.im_path=im_path
        self.ims=os.listdir(im_path)
        self.ims.sort()
        self.first_im=True
        self.rotations=np.arange(0,360,1)
        self.queen=np.load(kernel_path)

        self.masks = None
        self.queen_stack = None

        if (not queen_stack_path is None) and (not masks_path is None) :  
            self.masks = np.load(masks_path)
            self.queen_stack = np.load(queen_stack_path)
            print("[+] path for masks and rotated kernel is given")
        
        self.marker=None
        if not marker_path is None:
            self.marker=np.load(marker_path)            
            print("[+] Marker uploaded")

        
        self.fixed_queen_dim=self.queen.shape
        self.big_dim=[0,0]
        # self.big_dim[0]=int(np.sqrt(self.queen.shape[0]**2+self.queen.shape[1]**2))+1
        # self.big_dim[1]=int(np.sqrt(self.queen.shape[0]**2+self.queen.shape[1]**2))+1
        #! cheat 150
        self.big_dim[0]=150*2
        self.big_dim[1]=150*2
        self.queen_square=put_in_canvas(self.queen,self.big_dim,center=False)

    def loop(self):
        rot_track_stack=[]
        drift_stack=[]
        for c , im_name in enumerate( self.ims ):
            img=cv2.imread( os.path.join(self.im_path , im_name) )
            if len(img.shape)>2:
                img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            

            res , self.rot_track , drift = self.process_img(img)

            drift_stack.append(drift)
            rot_track_stack.append(self.rot_track)
            cv2.imwrite( os.path.join(self.out_path , "{:06d}.png".format(c)) , res)
            print("[+] progress {:.02f} %".format(c/len(self.ims)*100) , end='\r' )

            if c%100 == 0 or c == len(self.ims)-1:
                np.save(  os.path.join(self.out_path , "rot_tracks_{}.npy".format(CODE_START_TIME)) ,  np.array(rot_track_stack) )
                np.save(  os.path.join(self.out_path , "drift_stack_{}.npy".format(CODE_START_TIME)) ,  np.array(drift_stack) )

    def process_img(self,im):
        t=time()
        drift = None #! dift will be kept at None if no marker is uploaded
        if not self.marker is None:
            drift = find_marker(im,self.marker)

        if self.first_im: #! this is first frame so initilized kernel
            self.first_im=False
            init_pos_crop=im[ (im.shape[0]-CROP_WIDTH)//2:(im.shape[0]+CROP_WIDTH)//2 , (im.shape[1]-CROP_WIDTH)//2:(im.shape[1]+CROP_WIDTH)//2 ]

            self.queen , self.rot_track , self.queen_filt , self.queen_stack , self.masks =get_first_queen( init_pos_crop ,self.queen_square , self.big_dim , self.fixed_queen_dim,self.rotations ,self.masks , self.queen_stack)
            res= np.copy(self.queen_square) #! initilization to avoid type related error
            #! making rotation stacks tensor only for once 
            self.queen_stack=torch.tensor(self.queen_stack)
            self.queen_stack=self.queen_stack.reshape((1 , 1 , self.queen_stack.shape[0] , self.queen_stack.shape[1] , self.queen_stack.shape[2]))
            self.queen_stack=self.queen_stack.float()

        else: #! this is not first frame so proceed with tracking
          res,self.rot_track = crop_queen(im,self.queen,self.rot_track,self.rotations,CROP_WIDTH, self.queen_stack , self.masks , drift)
        print("[+] process_img duration {}".format(time()-t))
        return res , self.rot_track , drift
    
class Ros_Agent(Convolve_Agent):
    def __init__(self,sub_topic,pub_topic,im_path="." , out_path="." , kernel_path="." , queen_stack_path = None , masks_path = None , marker_path=None):
        super().__init__(im_path, out_path, kernel_path , queen_stack_path , masks_path , marker_path)
        self.sub_topic=sub_topic
        if "compressed" in self.sub_topic["im"]:
            self.im_sub=rospy.Subscriber(sub_topic["im"], CompressedImage, self.callback_in_im,queue_size=QUEUE_SIZE)
        else:
            self.im_sub=rospy.Subscriber(sub_topic["im"], Image, self.callback_in_im)
        if "pos" in self.sub_topic.keys():
            #! do this if beepos is supposed to be published
            self.pos_sub=rospy.Subscriber(sub_topic["pos"], BeePositionArray, self.callback_queen_position)

        
        self.bridge = CvBridge()

        #! for rospy default queue size is 10
        self.rot_track_publisher=rospy.Publisher(pub_topic["rot"], String, queue_size=QUEUE_SIZE)
        self.queen_crop_publisher=rospy.Publisher(pub_topic["queen"], Image, queue_size=QUEUE_SIZE)

        node_name=sys.argv[0].split("/")[-1].split(".")[0]
        rospy.init_node(node_name, anonymous=True)

    def callback_queen_position(self,data):
        if len(data.positions) > 0 :
            data.positions[0].alpha=self.rot_track
            self.added_rot_msg.publish(data)

    def callback_in_im(self,data):
        t=time()
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

        res , self.rot_track , drift = self.process_img(cv_image)
        self.rot_track_publish(self.rot_track)
        self.queen_crop_publish(res)

        print("[+] loop duration {}".format(time()-t))

    def rot_track_publish(self,data):
        self.rot_track_publisher.publish(str(data))

    def queen_crop_publish(self,data):
        data_cv2=self.bridge.cv2_to_imgmsg(data)
        self.queen_crop_publisher.publish(data_cv2)



def main():
    print("[+] code will run in DEVICE mode of: {}".format(DEVICE))
    if DEVICE == "ros": #! code will work as a rosnode
        args = rospy.myargv(argv=sys.argv)
        print("[+] passed arguments ",args)
        sub_topic={"im" : args[1]}
        pub_topic={"rot" : args[3] , "queen" : args[4]}
        
        essential_base_path=args[2]
        
        kernel_path = os.path.join(essential_base_path , 'used_kernel.npy')
        queen_stack_path = os.path.join(essential_base_path , 'queen_stack.npy')
        masks_path = os.path.join(essential_base_path , 'masks.npy')
        marker_path = os.path.join(essential_base_path , 'marker.npy')

        ros_agent=Ros_Agent(sub_topic,pub_topic, kernel_path=kernel_path , queen_stack_path = queen_stack_path , masks_path = masks_path , marker_path=marker_path)
        rospy.spin()
    else: #! code will process files
        if DEVICE == "thinkpad":
            im_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/rosbag2video"
            out_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0]+CODE_START_TIME+"3D_convolution"
            if not os.path.isdir(out_path):
                os.makedirs(out_path,exist_ok=True)    
            kernel_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/essentials/used_kernel.npy"
            queen_stack_path ="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/essentials/queen_stack.npy"
            masks_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/essentials/masks.npy"
            marker_path = None
        elif DEVICE == "server":
            im_path="/home/users/aamjadi/hdd/roboroyal/rosbag2video"
            out_path="/home/users/aamjadi/hdd/roboroyal/"+sys.argv[0].split("/")[-1].split(".")[0]+CODE_START_TIME+"3D_CONV_GPU"
            if not os.path.isdir(out_path):
                os.makedirs(out_path,exist_ok=True)    
            kernel_path="/home/users/aamjadi/hdd/roboroyal/essentials/used_kernel.npy"
            queen_stack_path ="/home/users/aamjadi/hdd/roboroyal/essentials/queen_stack.npy"
            masks_path="/home/users/aamjadi/hdd/roboroyal/essentials/masks.npy"
        agent=Convolve_Agent(im_path,out_path,kernel_path , queen_stack_path , masks_path ,marker_path)
        agent.loop()

if __name__=='__main__':
    main()

# import cProfile
# cProfile.run('main()')
