import glob
import cv2
import argparse
import numpy as np
import os
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--frame_dict', type=str, help='frame directory')
parser.add_argument('--output', type=str, help='output video')
parser.add_argument('--fps_vid', type=str, help='Video to take FPS from')
parser.add_argument('--h', type=int, help='output video height')
parser.add_argument('--w', type=int, help='output video width')
parser.add_argument('--use_vdims', help='use video dimensions', action="store_true")

args = parser.parse_args()

path = glob.glob(args.frame_dict+"*.png")

path.sort()

frame0 = cv2.imread(path[0])
fheight, fwidth, channels = frame0.shape

width = fwidth
height = fheight

cam = cv2.VideoCapture(args.fps_vid)
fps = cam.get(cv2.CAP_PROP_FPS)

if args.use_vdims:
	height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	width  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
	print("height: ", height, "| width:", width)

cam.release()

if args.w is not None and args.h is None:
	width = args.w
	height = int((width * fheight) / fwidth)

if args.h is not None and args.w is None:
	height = args.h
	width = int((height * fwidth) / fheight)

if args.h is not None and args.w is not None:
	height = args.h
	width = args.w

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

print("Writing Video with fps:", fps, ", height:", height, ", width", width)

for img in path:
	frame = cv2.imread(img)
	print("processing: ", img)
	resize = cv2.resize(frame,(width, height), interpolation = cv2.INTER_CUBIC)
	out.write(resize)

out.release()