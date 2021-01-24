import numpy as np
import argparse
import os
import cv2
from renderpose import *
import time

initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input_vid", type=str, help='input video file')
parser.add_argument("--output_vid", type=str, help='ouput video file')
parser.add_argument("--output_frames", type=str, help='ouput frame directory')
parser.add_argument("--display", help='display the output', action="store_true")
args = parser.parse_args()

if args.output_frames:
    print("Writing Frames")
    path = args.input_vid
    if args.output_vid:
        path = args.output_vid
    path_split = path.split("/")
    name = path_split[len(path_split)-1]
    name_split = name.split(".")
    output_dir = args.output_frames

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    print("Output frames disabled");

cap = cv2.VideoCapture(args.input_vid)
fps = cap.get(cv2.CAP_PROP_FPS)
height = 512
width  = 1024
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
 
if args.output_vid:
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output_vid, fourcc, fps, (width, height))

print("Height:", height, ", Width:", width, ", FPS:", fps, ", Frames:", frame_count)

fi = 0

while(cap.isOpened()):
    res, frame = cap.read()
    if res == True:
        scale_n, translate_n = resize_scale(frame)
        out_frame = fix_image(scale_n, translate_n, frame)
        if args.output_vid:
            out.write(out_frame)
        if args.output_frames:
            fcount = 000000000000 + fi
            _filename = name_split[0]+"_"+'{:0>12}'.format(fi)+".png"
            filename = os.path.join(args.output_frames, _filename)
            cv2.imwrite(filename, out_frame)
        if args.display:
            cv2.imshow("Output", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        nperc = int((fi/frame_count)*100.0)
        print("Progress: ",nperc,"%", end="\r")
        fi = fi + 1
    else:
        break
 
time_taken = time.time() - initTime
print(f"Process completed sucessfully, Total time taken: {time_taken}s, Rate: {frame_count/time_taken} FPS")
 
if args.output_vid:    
    out.release()
cap.release()

