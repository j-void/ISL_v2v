import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2
import time
import sys

sys.path.append("/Users/janmesh007/Documents/IIITB/ISL_v2v/")
#print(sys.path)
import util.hand_utils as hand_utils

initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--video", type=str, help="video input")
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

myshape = (1080, 1920, 3)

bbox_list = []

skip_index = 0

bbox_sizes = []

posepts_arr = np.zeros((25, 2))

cap = cv2.VideoCapture(args.video)

while(cap.isOpened()):
    res, frame = cap.read()
    if res == True:

        lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(frame, fix_coords=True)
        lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
        rbx, rby, rbw = hand_utils.assert_bbox(rfpts)

        bbox_sizes.append(lbw)
        bbox_sizes.append(rbw)
        #print(lbx, lby, lbw, rbx, rby, rbw)
        bbox_list.append([lbx, lby, lbw, rbx, rby, rbw])

        if args.display:
            frame = cv2.putText(frame, str(int(lbw)), (int(lbx+lbw/2), int(lby+lbw/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, str(int(rbw)), (int(rbx+rbw/2), int(rby+rbw/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (int(lbx), int(lby)), (int(lbx+lbw), int(lby+lbw)), (255, 0, 0), 1)
            frame = cv2.rectangle(frame, (int(rbx), int(rby)), (int(rbx+rbw), int(rby+rbw)), (255, 0, 0), 1)
            cv2.imshow("Skleton Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

print(f'Max hand bbox: {max(bbox_sizes)}')
     