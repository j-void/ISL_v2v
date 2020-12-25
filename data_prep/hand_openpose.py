import cv2
import numpy as np
import os
import glob
import argparse
from renderpose import *
import sys

sys.path.append("../util")
from hand_utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keypoints", type=str, help="keypoint directory")
parser.add_argument("--frame_dir", type=str, help='frame directory')
parser.add_argument("--input_vid", type=str, help='save directory')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

keypoint_path = os.path.join(args.keypoints, "*.json")
keypoints = glob.glob(keypoint_path)
keypoints.sort()

cap = cv2.VideoCapture(args.input_vid)
fps = cap.get(cv2.CAP_PROP_FPS)

fi = 0

while(cap.isOpened()):
    res, frame = cap.read()
    if res == True:
        scale_n, translate_n = resize_scale(frame)
        out_frame = fix_image(scale_n, translate_n, frame)
        posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[fi])
        xr, yr, wr = assert_bbox(get_keypoint_array(r_handpts))
        cv2.rectangle(out_frame, (xr, yr), (xr + wr, yr + wr), (255, 0, 0), 2)
        xl, yl, wl = assert_bbox(get_keypoint_array(l_handpts))
        cv2.rectangle(out_frame, (xl, yl), (xl + wl, yl + wl), (255, 0, 0), 2)
        fi = fi + 1
        cv2.imshow("Output", out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()