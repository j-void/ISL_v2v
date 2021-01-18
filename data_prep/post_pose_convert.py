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

args = parser.parse_args()

myshape = (1080, 1920, 3)

if args.save_dir:
    savedir = args.save_dir
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    if not os.path.exists(savedir + '/train_label'):
        os.makedirs(savedir + '/train_label')
    if not os.path.exists(savedir + '/train_img'):
        os.makedirs(savedir + '/train_img')
 
img_path = os.path.join(args.frame_dir, "*.png")
imgs = glob.glob(img_path)
imgs.sort()

keypoint_path = os.path.join(args.keypoints, "*.json")
keypoints = glob.glob(keypoint_path)
keypoints.sort()

print(f"Initialize -> Total frames: {len(imgs)}")

skip_index = 0

for f in range(len(imgs)):
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[f])
    if not posepts or not facepts:
        print("Skipping frame: ", f)
        skip_index = skip_index + 1
        continue
    _frame = cv2.imread(imgs[f])
    height, width, _ = _frame.shape
    output_frame = np.zeros((height, width, 3), np.uint8)
    output_frame.fill(255)
    #display_skleton(output_frame, posepts, facepts, r_handpts, l_handpts)
    if display_skleton(output_frame, posepts, facepts, r_handpts, l_handpts) == False:
        print("Skipping frame: ", f)
        skip_index = skip_index + 1
        continue
    if args.save_dir:
        _fn = "frame_"+'{:0>12}'.format(f)+".png"
        _filepath_label = os.path.join(savedir, "train_label")
        filename_label = os.path.join(_filepath_label, _fn)
        _filepath_img = os.path.join(savedir, "train_img")
        filename_img = os.path.join(_filepath_img, _fn)
        print("Processing frame: ", f)
        cv2.imwrite(filename_label, output_frame)
        cv2.imwrite(filename_img, _frame)
    if args.display:
        cv2.imshow("Skleton Frame", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
print(f"Summary -> Total frames: {len(imgs)}, Processed frames: {len(imgs) - skip_index}, Skipped: {skip_index}")
     