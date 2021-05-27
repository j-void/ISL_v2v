import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2
import time
import pickle
import sys

sys.path.append(os.getcwd())
print(sys.path)
import util.hand_utils as hand_utils
initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keypoints", type=str, help="keypoint directory")
parser.add_argument("--frame_dir", type=str, help='frame directory')
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--vid", type=str, help='original video')
parser.add_argument("--out_keypoints_pkl" ,type=str, help='output pkl for transformed keypoints')
parser.add_argument("--train_dir", type=str, help='train directory')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

myshape = (1080, 1920, 3)

cap1 = cv2.VideoCapture(args.vid)
fps1 = cap1.get(cv2.CAP_PROP_FPS)
frame_count1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
cap1.release()

bbox_list = []

posepts_list = []
facepts_list = []
r_handpts_list = []
l_handpts_list = []

if args.save_dir:
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

train_pose_path = os.path.join(args.train_dir, "avg_pose.txt")
avg_train_pose = np.loadtxt(train_pose_path)

import joblib
max_bbox_size = joblib.load(os.path.join(args.train_dir, "bbox_out.pkl"))["max_bbox"]

print(f"Initialize -> Total frames: {len(imgs)}")

skip_index = 0

bbox_sizes = []

enb = False

prev_pparr = np.array([0.0, 0.0])
prev_scale = 1.0

posepts_arr = np.zeros((25,2))

for f in range(len(imgs)):
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[f])
    if not posepts or not facepts:
        skip_index = skip_index + 1
        continue
    posepts_arr = posepts_arr + get_keypoint_array_pose(posepts)

posepts_arr = posepts_arr / (len(imgs) - skip_index)

skip_index = 0

dist_test = np.linalg.norm(posepts_arr[5] - posepts_arr[2])
dist_train = np.linalg.norm(avg_train_pose[5] - avg_train_pose[2])
scale = dist_train/dist_test

posepts_arr_t = apply_transformation_arr(posepts_arr, (0.0, 0.0), scale)

zero_tf = [0.0, 0.0, 0.0]

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

    posepts = apply_transformation(posepts, (0.0, 0.0), scale)
    facepts = apply_transformation(facepts, (0.0, 0.0), scale)
    r_handpts = apply_transformation(r_handpts, (0.0, 0.0), scale)
    l_handpts = apply_transformation(l_handpts, (0.0, 0.0), scale)
    zero_tf = apply_transformation(zero_tf, (0.0, 0.0), scale)
    
    #_frame = fix_image(scale, (0.0, 0.0), _frame)
    _frame = cv2.resize(_frame, (int(_frame.shape[1] * scale), int(_frame.shape[0] * scale)))
            
    translation = (-posepts_arr_t[1,0] + avg_train_pose[1,0], -posepts_arr_t[1,1] + avg_train_pose[1,1])
    
    posepts = apply_transformation(posepts, translation, 1.0)
    facepts = apply_transformation(facepts, translation, 1.0)
    r_handpts = apply_transformation(r_handpts, translation, 1.0)
    l_handpts = apply_transformation(l_handpts, translation, 1.0)
    zero_tf = apply_transformation(zero_tf, translation, 1.0)
    
    _frame = fix_image(1.0, translation, _frame)
    
    body_pose_arr = get_keypoint_array_pose(posepts)
    translation_face = (-body_pose_arr[0,0] + avg_train_pose[0,0], -body_pose_arr[0,1] + avg_train_pose[0,1])
    facepts = apply_transformation(facepts, translation_face, 1.0)
    posepts[0:2] = apply_transformation(posepts[0:2], translation_face, 1.0)
    posepts[45:56] = apply_transformation(posepts[45:56], translation_face, 1.0)
    #out_frame = fix_image(1.3, (0.0, -40.0), frame)
    #display_skleton(output_frame, posepts, facepts, r_handpts, l_handpts)
    
    posepts_list.append(posepts)
    facepts_list.append(facepts)
    r_handpts_list.append(r_handpts)
    l_handpts_list.append(l_handpts)
    
    if display_skleton(output_frame, posepts, facepts, r_handpts, l_handpts, [0,0]) == False:
        print("Skipping frame: ", f)
        skip_index = skip_index + 1
        continue
    
    enb = False
    
    lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(_frame, fix_coords=True)
    lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
    rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
    if check_detected(r_handpts) ==  False:
        rbw = 0
    if check_detected(l_handpts) == False:
        lbw = 0
    bbox_sizes.append(lbw)
    bbox_sizes.append(rbw)
    
    bbox_list.append([lbx, lby, lbw, rbx, rby, rbw])

    posepts_arr = posepts_arr + get_keypoint_array_pose(posepts)
    
    if args.save_dir:
        _fn = "frame_"+'{:0>12}'.format(f)+".png"
        _filepath_label = os.path.join(savedir, "test_label")
        filename_label = os.path.join(_filepath_label, _fn)
        _filepath_img = os.path.join(savedir, "test_img")
        filename_img = os.path.join(_filepath_img, _fn)
        print("Processing frame: ", f)
        keypoints_dict = {'max_bbox' : max_bbox_size, 'bbox_list':bbox_list}
        outfile = open(os.path.join(savedir, "bbox_out.pkl"), 'wb')
        import pickle
        pickle.dump(keypoints_dict, outfile)
        outfile.close()
        cv2.imwrite(filename_label, output_frame)
        cv2.imwrite(filename_img, _frame)
    if args.display:
        _output = cv2.hconcat((_frame, output_frame))
        cv2.imshow("Skleton Frame", _output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

if args.out_keypoints_pkl:
    keypoints_dict = {'posepts' : posepts_list, 'facepts':facepts_list, 'r_handpts':r_handpts_list, 'l_handpts':l_handpts_list, 'fps': fps1}
    outfile = open(args.out_keypoints_pkl, 'wb')
    pickle.dump(keypoints_dict, outfile)
    outfile.close()

time_taken = time.time() - initTime
print(f"Summary -> Total frames: {len(imgs)}, Processed frames: {len(imgs) - skip_index}, Skipped: {skip_index}, Total time taken: {time_taken}s, Rate: {(len(imgs) - skip_index)/time_taken} FPS")
     