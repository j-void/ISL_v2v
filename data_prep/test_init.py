import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keypoints", type=str, help="keypoint directory")
parser.add_argument("--frame_dir", type=str, help='frame directory')
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--display", help='display the output', action="store_true")
parser.add_argument("--no_save", help='not save the output', action="store_true")

args = parser.parse_args()

myshape = (1080, 1920, 3)
savedir = args.save_dir

if not os.path.exists(savedir):
    	os.makedirs(savedir)
if not os.path.exists(savedir + '/test_label'):
	os.makedirs(savedir + '/test_label')
if not os.path.exists(savedir + '/test_img'):
	os.makedirs(savedir + '/test_img')
 
img_path = os.path.join(args.frame_dir, "*.png")
imgs = glob.glob(img_path)
imgs.sort()

keypoint_path = os.path.join(args.keypoints, "*.json")
keypoints = glob.glob(keypoint_path)
keypoints.sort()

for f in range(len(imgs)):
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[f])
    _frame = cv2.imread(imgs[f])
    height, width, _ = _frame.shape
    output_frame = np.zeros((height, width, 3), np.uint8)
    output_frame.fill(255)
    display_skleton(output_frame, posepts, facepts, r_handpts, l_handpts)
    scale_n, translate_n = resize_scale(output_frame)
    out_sk = fix_image(scale_n, translate_n,output_frame)
    out_frame = fix_image(scale_n, translate_n,_frame)
    _fn = "frame_"+'{:0>12}'.format(f)+".png"
    _filepath_label = os.path.join(savedir, "test_label")
    filename_label = os.path.join(_filepath_label, _fn)
    _filepath_img = os.path.join(savedir, "test_img")
    filename_img = os.path.join(_filepath_img, _fn)
    print("Processing frame: ", f)
    if not args.no_save:
        cv2.imwrite(filename_label, out_sk)
        cv2.imwrite(filename_img, out_frame)
    if args.display:
        cv2.imshow("Skleton Frame", out_sk)
     