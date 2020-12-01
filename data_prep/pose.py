import numpy as np
import cv2
import glob
import os
import argparse
from renderpose import *
import time


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keypoints", type=str, help="keypoint directory")
parser.add_argument("--vid", type=str, help='video file')

args = parser.parse_args()

keypoint_path = os.path.join(args.keypoints, "*.json")

keypoints = glob.glob(keypoint_path)
keypoints.sort()
fi = 0

cap = cv2.VideoCapture(args.vid)


while(cap.isOpened()):
    res, frame = cap.read()
    sk_frame = frame.copy()
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[fi])
    display_keypoints(frame, posepts, facepts, r_handpts, l_handpts)
    display_skleton(sk_frame, posepts, facepts, r_handpts, l_handpts)
    cv2.imshow("Frame", frame)
    cv2.imshow("Skleton Frame", sk_frame)
    fi = fi + 1
    if fi >= len(keypoints):
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()




