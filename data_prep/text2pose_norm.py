import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2
import time
import joblib

initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--input_kpts_pkl",  nargs='+', type=str, help='input video file')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

if args.save_dir:
    savedir = args.save_dir
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    if not os.path.exists(savedir + '/test_label'):
        os.makedirs(savedir + '/test_label')

    
keypoints_pkls = args.input_kpts_pkl
keypoints_all = []

index_count = 0

for ki in range(len(keypoints_pkls)-1):
    keypoints = joblib.load(keypoints_pkls[ki])
    frame_count = len(keypoints["posepts"])

    fps = keypoints["fps"]
    
    j = 4

    k1_ypts = []
    k1_xpts = []

    for i in range(len(keypoints["posepts"])):
        k1_xpts.append(keypoints["posepts"][i][j*3])
        k1_ypts.append(keypoints["posepts"][i][j*3+1])

    #pickup end

 
time_taken = time.time() - initTime           
print (f"Total frames: {index_count}, time taken: {time_taken}, Rate: {index_count/time_taken}FPS")
    