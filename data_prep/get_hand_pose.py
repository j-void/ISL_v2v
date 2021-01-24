import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2
import time
import joblib

import sys

sys.path.append('util')
print(sys.path)

import hand_utils as hand_utils

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
    real_img = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
    scale_n, translate_n = hand_utils.resize_scale(real_img)
    real_img = hand_utils.fix_image(scale_n, translate_n, real_img)
    lhpts_real, rhpts_real, hand_state_real = hand_utils.get_keypoints_holistic(real_img, fix_coords=True)
    lhsk_real = np.zeros((128, 128, 3), dtype=np.uint8)
    lhsk_real.fill(255)
    rhsk_real = np.zeros((128, 128, 3), dtype=np.uint8)
    rhsk_real.fill(255)
    hand_utils.display_single_hand_skleton(lhsk_real, lhpts_real)
    hand_utils.display_single_hand_skleton(rhsk_real, rhpts_real)
    real_img = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
    hand_utils.display_hand_skleton(real_img, rhpts_real, lhsk_real)
    
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[f])
    if not r_handpts and not l_handpts:
        hand_utils.display_hand_skleton(_frame, get_keypoint_array(r_handpts), get_keypoint_array(l_handpts))
    
    
    if args.display:
        cv2.imshow("Openpose SK", _frame)
        cv2.imshow("Mediapipe SK", real_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

