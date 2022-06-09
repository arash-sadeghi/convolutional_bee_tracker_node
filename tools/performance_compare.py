import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
from time import time , ctime
CODE_BEGIN_TIME = ctime(time()).replace(":","_").replace(" ","_")
def integrate2onefile(in_path):
    files=os.listdir(in_path)
    integrated=[]
    for f in files:
        data=np.loadtxt(os.path.join(in_path,f),delimiter=',')
        angle=data[3]
        row=int(f.split(".")[0])
        integrated.append([row , angle])
    return np.array(integrated)

def cast_to_minuspi2pi(inp):
    while(inp>=180):
        inp=inp-360
    while(inp<=-180):
        inp=inp+360
    return inp

def get_cdf(array,bins):
    array=np.abs(array)
    count, bins_count = np.histogram(array,bins=bins)
    pdf = count / count.sum()
    cdf = np.cumsum(pdf)
    return [cdf,bins_count[1:]]

def get_errors(performance , gts_):
    #! correcting gt
    gts_[:,1]= - gts_[:,1] + 360

    related_performance = performance[(gts[:,0]).astype(np.int64)]

    errors=[]
    for i in range(len(related_performance)):
        related_performance[i]=cast_to_minuspi2pi(related_performance[i])
        gts_[i,1]=cast_to_minuspi2pi(gts_[i,1])
        tmp = cast_to_minuspi2pi(related_performance[i]-gts_[i,1])
        errors.append(tmp)
    return errors

if __name__ == '__main__':

    gt_folder_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/DenseObjectAnnotation/static/txt"
    # performance_file_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/without_marker_tracking/rot_tracks_Sun_Apr__3_19_11_11_2022.npy"
    performance_file_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/ConAndRep_fused3D_convolution/rot_tracks_Tue_Apr__5_15_52_32_2022.npy"
    performance=np.load(performance_file_path)
    gts=integrate2onefile(gt_folder_path)
    errors=get_errors(performance , np.copy(gts)) #! !!! if you dont do np.copy(), the gts inside main function will change as well
    plt.hist(errors,bins=200)
    plt.ylim(0,20)
    plt.grid()

    plt.show()
    exit(0)
    cdf , bins = get_cdf(errors,200)
    plt.plot(bins,cdf,label='without marker tracking')

    performance_file_path="/home/arash/catkin_ws/src/convolutional_bee_tracker/data/rot_tracks_Sun_Apr__3_13_29_55_2022_from_server.npy"
    performance=np.load(performance_file_path)
    errors=get_errors(performance , np.copy(gts))
    # plt.hist(errors,bins=200)
    # plt.ylim(0,20)
    # plt.grid()
    cdf , bins = get_cdf(errors,200)
    plt.plot(bins,cdf,label='with marker tracking')
    plt.title("method comparison")
    plt.legend()
    plt.xlabel('Absolute Value of Rotation Error [deg]')
    plt.ylabel('Prob. of Correct Rotation Estimation')
    plt.grid()
    # plt.savefig('performance of {} .png'.format( performance_file_path.split("/")[-1].split(".")[0] ))
    plt.savefig('/home/arash/catkin_ws/src/convolutional_bee_tracker/data/cdf_of_method_comparison_{}_.png'.format(CODE_BEGIN_TIME))
    plt.show()
    print("[+] Done~")
    
    

