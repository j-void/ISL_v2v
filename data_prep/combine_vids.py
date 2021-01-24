import numpy as np
import argparse
import os
import cv2
from renderpose import *
import time

initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input_vid",  nargs='+', type=str, help='input video file')
parser.add_argument("--output_vid", type=str, help='ouput video file')
parser.add_argument("--output_frames", type=str, help='ouput frame directory')
parser.add_argument("--display", help='display the output', action="store_true")
args = parser.parse_args()

vids = args.input_vid
cap = cv2.VideoCapture(vids[0])
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if args.output_vid:
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output_vid, fourcc, fps, (1024, 512))



for v in range(len(vids)):
    cap = cv2.VideoCapture(vids[v])
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fi = 0
    while (cap.isOpened()):
        res, frame = cap.read()
        if res == True:
            scale_n, translate_n = resize_scale(frame)
            out_frame = fix_image(scale_n, translate_n, frame)
            if args.output_vid:
                out.write(out_frame)
            if args.output_frames:
                fcount = 000000000000 + fi
                _filename = "frame_"+'{:0>12}'.format(fi)+".png"
                filename = os.path.join(args.output_frames, _filename)
                cv2.imwrite(filename, out_frame)
            if args.display:
                cv2.imshow("Output", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            nperc = int((fi/frame_count)*100.0)
            print(f"Progress: ({v+1}/{len(vids)}) - {nperc}%", end="\r")
            fi = fi + 1
        else:
            break

time_taken = time.time() - initTime
print(f"Process completed sucessfully, Total time taken: {time_taken}s")
 
if args.output_vid:    
    out.release()
cap.release()
            
            