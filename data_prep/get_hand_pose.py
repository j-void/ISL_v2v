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
parser.add_argument("--keypoints", type=str, help="keypoint directory")
parser.add_argument("--frame_dir", type=str, help='frame directory')
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--ith_pkl", type=str, help='pkl file for interhand output')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

img_path = os.path.join(args.frame_dir, "*.png")
imgs = glob.glob(img_path)
imgs.sort()

keypoint_path = os.path.join(args.keypoints, "*.json")
keypoints = glob.glob(keypoint_path)
keypoints.sort()

ithkp = joblib.load(args.ith_pkl)

hand_pose = ithkp["hand_pose"]

print(f"Initialize -> Total frames: {len(imgs)}")

for f in range(len(imgs)):
    _frame = cv2.imread(imgs[f])
    display_single_hand_skleton_left(_frame, hand_pose[f])
    display_single_hand_skleton_right(_frame, hand_pose[f])
    if args.display:
        cv2.imshow("Skleton Frame", _frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

