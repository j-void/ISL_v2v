import numpy as np
import argparse
import os
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--video", type=str, help='video file')
parser.add_argument("--output", type=str, help='ouput directory')

args = parser.parse_args()

path = args.video
path_split = path.split("/")
name = path_split[len(path_split)-1]
name_split = name.split(".")
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(args.video)

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('frames count:', frame_count)

i=0

while(cap.isOpened()):
    res, frame = cap.read()
    if res == True:
        fcount = 000000000000 + i
        _filename = name_split[0]+"_"+'{:0>12}'.format(i)+".png"
        filename = os.path.join(args.output, _filename) 
        print("Writing: ", filename)
        cv2.imwrite(filename, frame)
        i = i + 1
    else:
        break
    
cap.release()