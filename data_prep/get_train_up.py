import cv2 as cv 
import numpy as np
import json
import os
import argparse
from renderopenpose import *
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--keypoints_dir', type=str, default='keypoints', help='directory where target keypoint files are stored, assumes .yml format for now.')
parser.add_argument('--frames_dir', type=str, default='frames', help='directory where source frames are stored. Assumes .png files for now.')
parser.add_argument('--save_dir', type=str, default='save', help='directory where to save generated files')

opt = parser.parse_args()

myshape = (1080, 1920, 3)

keypoints_dir = opt.keypoints_dir 
frames_dir = opt.frames_dir 
savedir = opt.save_dir

keypoints_path = os.path.join(keypoints_dir, "*json")
keypoints = glob.glob(keypoints_path)
keypoints.sort()

if not os.path.exists(savedir):
	os.makedirs(savedir)