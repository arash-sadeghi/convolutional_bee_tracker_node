#! /usr/bin/env python
import os
import sys #! not added to dependencies
import pdb #! not added to dependencies
import numpy as np #! not added to dependencies
import cv2 #! not added to dependencies
# from cv2 import imwrite, equalizeHist , matchTemplate , TM_CCORR_NORMED , COLOR_BGR2GRAY , cvtColor #! not added to dependencies
import matplotlib.pyplot as plt #! not added to dependencies
from scipy.ndimage.interpolation import rotate #! not added to dependencies
from time import ctime , time
# from torch.nn import functional as F
# import torch 
# def manual_conv2d(inp,ker):
#     inp=torch.tensor(im_cropped)
#     inp=inp.reshape((1,1,inp.shape[0],inp.shape[1]))
#     ker=torch.tensor(queen-np.mean(queen[:]))
#     ker=ker.reshape((1,1,ker.shape[0],ker.shape[1]))
#     out=F.conv2d(inp,ker)
#     return out[0,0,:,:].numpy()
CODE_START_TIME = ctime(time()).replace(":","_").replace(" ","_")
QUEUE_SIZE=1
# CROP_WIDTH=400
#! cheat 300
CROP_WIDTH=300
MARKER_CROP_WIDTH=300
MARKER_DISPALCEMENT_WARNING_THRESH=100
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

    #! for relative rots:
    # rot_track+=rotations[chosen_rot]

    #! for absolute rots:
    rot_track = np.copy(rotations[chosen_rot])

    #! getting the position of queen based on heatmap
    indx=np.unravel_index(matched.argmax(),matched.shape)
    #! updating queen position also will track position change of queen.
    #! which is not needed in our case since queen is always in the middle
    res=im_cropped[indx[0]:indx[0]+queen_stack.shape[1] , indx[1]:indx[1]+queen_stack.shape[2]]

    res[np.round(masks[chosen_rot])<=0]=0
    # queen=cv2.equalizeHist(queen)
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
        filt=np.round(masks[c])
        kernel=np.copy(queen_stack[c])
        kernel[filt==1]=kernel[filt==1]-np.mean(kernel[filt==1])
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
    def __init__(self,im_path,out_path,kernel_path , queen_stack_path = None , masks_path = None , marker_path=None):
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
        self.drift=None
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
            
            if not self.marker is None:
              self.drift = find_marker(img,self.marker)
              drift_stack.append(self.drift)

            res,self.rot_track = self.callback_in_im(img)
            rot_track_stack.append(self.rot_track)
            cv2.imwrite( os.path.join(self.out_path , "{:06d}.png".format(c)) , res)
            print("[+] progress {:.02f} %".format(c/len(self.ims)*100) , end='\r' )

            if c%100 == 0 or c == len(self.ims)-1:
                np.save(  os.path.join(self.out_path , "rot_tracks_{}.npy".format(CODE_START_TIME)) ,  np.array(rot_track_stack) )
                np.save(  os.path.join(self.out_path , "drift_stack_{}.npy".format(CODE_START_TIME)) ,  np.array(drift_stack) )


    def callback_in_im(self,im):
        if self.first_im: #! this is first frame so initilized kernel
            self.first_im=False
            init_pos_crop=im[ (im.shape[0]-CROP_WIDTH)//2:(im.shape[0]+CROP_WIDTH)//2 , (im.shape[1]-CROP_WIDTH)//2:(im.shape[1]+CROP_WIDTH)//2 ]

            self.queen , self.rot_track , self.queen_filt , self.queen_stack , self.masks =get_first_queen( init_pos_crop ,self.queen_square , self.big_dim , self.fixed_queen_dim,self.rotations ,self.masks , self.queen_stack)
            res=0
        
        else: #! this is not first frame so proceed with tracking
            res,self.rot_track = crop_queen(im,self.queen,self.rot_track,self.rotations,CROP_WIDTH, self.queen_stack , self.masks , self.drift)
            
        return res,self.rot_track
    



if __name__=='__main__':
    device="server"
    if device == "thinkpad":
      im_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/rosbag2video"
      out_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/"+sys.argv[0].split("/")[-1].split(".")[0]
      if not os.path.isdir(out_path):
          os.makedirs(out_path,exist_ok=True)    
      kernel_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/used_kernel.npy"
      queen_stack_path ="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/queen_stack.npy"
      masks_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/masks.npy"
      marker_path= None # "/home/arash/catkin_ws/src/convolutional_bee_tracker/data/marker.npy"
    elif device == "server":
      im_path="/home/users/aamjadi/hdd/roboroyal/rosbag2video"
      out_path="/home/users/aamjadi/hdd/roboroyal/"+sys.argv[0].split("/")[-1].split(".")[0]+"without_marker_tracking"
      if not os.path.isdir(out_path):
          os.makedirs(out_path,exist_ok=True)    
      kernel_path="/home/users/aamjadi/hdd/roboroyal/essentials/used_kernel.npy"
      queen_stack_path ="/home/users/aamjadi/hdd/roboroyal/essentials/queen_stack.npy"
      masks_path="/home/users/aamjadi/hdd/roboroyal/essentials/masks.npy"
      marker_path=None # "/home/users/aamjadi/hdd/roboroyal/essentials/marker.npy"
    agent=Convolve_Agent(im_path,out_path,kernel_path , queen_stack_path , masks_path ,marker_path)
    agent.loop()
