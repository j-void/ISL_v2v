import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2
import time
import joblib
import matplotlib.pyplot as plt
from scipy import interpolate
import rdp as rdp

initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--k1_pkl", type=str, help="word1 directory")
parser.add_argument("--k2_pkl", type=str, help='word2 directory')
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--train_dir", type=str, help='train directory')
parser.add_argument("--out_pkl", type=str, help='train directory')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

diff = 0.2
t1 = 0.2
t2 = 0.14

if args.save_dir:
    savedir = args.save_dir
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    if not os.path.exists(savedir + '/test_label'):
        os.makedirs(savedir + '/test_label')

keypoints1 = joblib.load(args.k1_pkl)
keypoints2 = joblib.load(args.k2_pkl)

posepts_list = keypoints1["posepts"] + keypoints2["posepts"]
facepts_list = keypoints1["facepts"] + keypoints2["facepts"]
r_handpts_list = keypoints1["r_handpts"] + keypoints2["r_handpts"]
l_handpts_list = keypoints1["l_handpts"] + keypoints2["l_handpts"]

posepts_list_new = []
facepts_list_new = []
r_handpts_list_new = []
l_handpts_list_new = []


frame_count1 = len(keypoints1["posepts"])
frame_count2 = len(keypoints2["posepts"])

fps1 = keypoints1["fps"]
fps2 = keypoints2["fps"]

avg_fps = (fps1+fps2)/2
print(avg_fps)

s1 = int(t1*frame_count1)
s2 = int(t2*frame_count2)

#pickup
j = 4

k1_ypts = []
k1_xpts = []

for i in range(len(keypoints1["posepts"])):
    k1_xpts.append(keypoints1["posepts"][i][j*3])
    k1_ypts.append(keypoints1["posepts"][i][j*3+1])


k2_ypts = []
k2_xpts = []

for i in range(len(keypoints2["posepts"])):
    k2_xpts.append(keypoints2["posepts"][i][j*3])
    k2_ypts.append(keypoints2["posepts"][i][j*3+1])

startpt = 0

for val in k1_ypts[::-1]:
    if val < 0:
        continue
    if (val - 416) < 0:
        pt = k1_ypts.index(val)
        startpt = pt
        plt.plot(pt, k1_ypts[pt],'r*')
        break
   
endpt = 0 
for val in k2_ypts:
    if val < 100:
        continue
    if (val - 416) < 0:
        pt = k2_ypts.index(val)-1
        endpt = pt
        plt.plot(pt+len(k1_ypts), k2_ypts[pt],'r*')
        break

#pickup end

#RDP start

_k1_ypts = []
_k1_ypts_list = []
for i in range(len(k1_ypts)):
    if k1_ypts[i] > 0:
        _k1_ypts.append(k1_ypts[i])
        _k1_ypts_list.append([i, k1_ypts[i]])

        
_k2_ypts = []
_k2_ypts_list = []
skip_ky2 = 0
for i in range(len(k2_ypts)):
    if k2_ypts[i] > 0:
        _k2_ypts.append(k2_ypts[i])
        _k2_ypts_list.append([i+len(k1_ypts), k2_ypts[i]])


min_angle = np.pi*0.22

simplified_trajectory_k1y = np.array(rdp.rdp(_k1_ypts_list, epsilon=10))
st_k1y, sy_k1 = simplified_trajectory_k1y.T
directions_k1y = np.diff(simplified_trajectory_k1y, axis=0)
theta_k1y = rdp.angle(directions_k1y)
idx_k1y = np.where(theta_k1y>min_angle)[0]+1

if startpt == 0:
    startpt = np.where(k1_ypts==sy_k1[idx_k1y[-1]])[0][0]

simplified_trajectory_k2y = np.array(rdp.rdp(_k2_ypts_list, epsilon=10))
st_k2y, sy_k2 = simplified_trajectory_k2y.T
directions_k2y = np.diff(simplified_trajectory_k2y, axis=0)
theta_k2y = rdp.angle(directions_k2y)
idx_k2y = np.where(theta_k2y>min_angle)[0]+1


_k1_xpts = []
_k1_xpts_list = []
for i in range(len(k1_xpts)):
    if k1_xpts[i] > 100:
        _k1_xpts.append(k1_xpts[i])
        _k1_xpts_list.append([i, k1_xpts[i]])

        
_k2_xpts = []
_k2_xpts_list = []
for i in range(len(k2_xpts)):
    if k2_ypts[i] > 100:
        _k2_xpts.append(k2_xpts[i])
        _k2_xpts_list.append([i+len(k1_xpts), k2_xpts[i]])



min_angle = np.pi*0.22

simplified_trajectory_k1x = np.array(rdp.rdp(_k1_xpts_list, epsilon=10))
st_k1x, sx_k1 = simplified_trajectory_k1x.T
directions_k1x = np.diff(simplified_trajectory_k1x, axis=0)
theta_k1x = rdp.angle(directions_k1x)
idx_k1x = np.where(theta_k1x>min_angle)[0]+1


simplified_trajectory_k2x = np.array(rdp.rdp(_k2_xpts_list, epsilon=10))
st_k2x, sx_k2 = simplified_trajectory_k2x.T
directions_k2x = np.diff(simplified_trajectory_k2x, axis=0)
theta_k2x = rdp.angle(directions_k2x)
idx_k2x = np.where(theta_k2x>min_angle)[0]+1

k1_all_dist = []
k1_all_dist_t = []
_dist_k1 = []
k1_z_dist = []
for i in range(1, len(k1_ypts)):
    if k1_ypts[i] > 0 and k1_ypts[i-1] > 0 and k1_xpts[i] > 100 and k1_xpts[i-1] > 100:
        _dist = np.linalg.norm(np.array([k1_xpts[i-1], k1_ypts[i-1]]) - np.array([k1_xpts[i], k1_ypts[i]]))
        if _dist == 0:
            k1_z_dist.append(i)
        k1_all_dist.append(_dist)
        k1_all_dist_t.append(i)
        _dist_k1.append([i, _dist])
        
k2_all_dist = []
k2_all_dist_t = []
_dist_k2 = []
k2_z_dist = []
for i in range(1, len(k2_ypts)):
    if k2_ypts[i] > 0 and k2_ypts[i-1] > 0 and k2_xpts[i] > 100 and k2_xpts[i-1] > 100:
        _dist = np.linalg.norm(np.array([k2_xpts[i-1], k2_ypts[i-1]]) - np.array([k2_xpts[i], k2_ypts[i]]))
        if _dist == 0:
            k2_z_dist.append(i+len(k1_ypts))
        k2_all_dist.append(_dist)
        k2_all_dist_t.append(i+len(k1_ypts))
        _dist_k2.append([i+len(k1_ypts), _dist])
          

simplified_trajectory_k1d = np.array(rdp.rdp(_dist_k1, epsilon=5))
st_k1d, sd_k1 = simplified_trajectory_k1d.T
directions_k1d = np.diff(simplified_trajectory_k1d, axis=0)
theta_k1d = rdp.angle(directions_k1d)
idx_k1d = np.where(theta_k1d>min_angle)[0]+1

for i in range(len(idx_k1d), 0, -1):
    if abs(sd_k1[i-1] - sd_k1[i]) < 3:
        _tmpIdx = np.where(_dist_k1==sd_k1[i-1])[0][0]
        if _tmpIdx < startpt:
            if int(st_k1d[i-1]) > 5 and len(idx_k1d)-i < 4:
                print("Considering 0 displacement value", _tmpIdx)
                startpt = int(st_k1d[i-1])
        break



simplified_trajectory_k2d = np.array(rdp.rdp(_dist_k2, epsilon=5))
st_k2d, sd_k2 = simplified_trajectory_k2d.T
directions_k2d = np.diff(simplified_trajectory_k2d, axis=0)
theta_k2d = rdp.angle(directions_k2d)
idx_k2d = np.where(theta_k2d>min_angle)[0]+1


#RDP end
_yymax = 480
#startpt = np.where(k1_xpts==sx_k1[idx_k1x[-2]])[0][0]
timspace = np.arange(startpt, endpt+frame_count1)
print(startpt, endpt+frame_count1)
 
for j in range(25):
    xrev_list = []
    yrev_list = []
    trev_list = []

    skp_this = False

    for i in range(startpt-10, startpt):
        if posepts_list[i][j*3] <= 0 or posepts_list[i][j*3 + 1] <= 0:
            skp_this = True
            continue
        trev_list.append(i)
        xrev_list.append(posepts_list[i][j*3])
        yrev_list.append(posepts_list[i][j*3 + 1])
        # if posepts_list[i][j*3 + 1] >= _yymax:
        #     yrev_list.append(_yymax)
        # else:
        #     yrev_list.append(posepts_list[i][j*3 + 1])
        
    for i in range(endpt+int(frame_count1), endpt+int(frame_count1)+10):
        if posepts_list[i][j*3] <= 0 or posepts_list[i][j*3 + 1] <= 0:
            skp_this = True
            continue
        trev_list.append(i)
        xrev_list.append(posepts_list[i][j*3])
        yrev_list.append(posepts_list[i][j*3 + 1])
        # if posepts_list[i][j*3 + 1] >= _yymax:
        #     yrev_list.append(_yymax)
        # else:
        #     yrev_list.append(posepts_list[i][j*3 + 1])
    
    if skp_this == True:
        skp_this = False
        continue
        
    zx_rev = np.polyfit(np.array(trev_list), np.array(xrev_list), 3)
    px_rev = np.poly1d(zx_rev)
    zy_rev = np.polyfit(np.array(trev_list), np.array(yrev_list), 3)
    py_rev = np.poly1d(zy_rev)
    xnew_rev = px_rev(timspace)
    ynew_rev = py_rev(timspace) 
    
    idx = 0
    for i in range(startpt, int(frame_count1+endpt)):
        posepts_list[i][j*3] = xnew_rev[idx]
        posepts_list[i][j*3+1] = ynew_rev[idx]
        #posepts_list_new.append([xnew_rev[idx], ynew_rev[idx], 0.5])
        idx = idx+1
    
        
for j in range(21):
    xrev_list = []
    yrev_list = []
    trev_list = []
    
    skp_this = False

    for i in range(startpt-10, startpt):
        if r_handpts_list[i][j*3] <= 0 or r_handpts_list[i][j*3 + 1] <= 0:
            skp_this = True
            continue
        trev_list.append(i)
        xrev_list.append(r_handpts_list[i][j*3])
        yrev_list.append(r_handpts_list[i][j*3 + 1])
        # if posepts_list[i][j*3 + 1] >= _yymax:
        #     yrev_list.append(_yymax)
        # else:
        #     yrev_list.append(r_handpts_list[i][j*3 + 1])
        
        
    for i in range(endpt+int(frame_count1), endpt+int(frame_count1)+10):
        if r_handpts_list[i][j*3] <= 0 or r_handpts_list[i][j*3 + 1] <= 0:
            skp_this = True
            continue
        trev_list.append(i)
        xrev_list.append(r_handpts_list[i][j*3])
        yrev_list.append(r_handpts_list[i][j*3 + 1])
        # if posepts_list[i][j*3 + 1] >= _yymax:
        #     yrev_list.append(_yymax)
        # else:
        #     yrev_list.append(r_handpts_list[i][j*3 + 1])
        
    if skp_this == True:
        skp_this = False
        continue
        
    zx_rev = np.polyfit(np.array(trev_list), np.array(xrev_list), 3)
    px_rev = np.poly1d(zx_rev)
    zy_rev = np.polyfit(np.array(trev_list), np.array(yrev_list), 3)
    py_rev = np.poly1d(zy_rev)
    xnew_rev = px_rev(timspace)
    ynew_rev = py_rev(timspace) 
    
    idx = 0
    for i in range(startpt, int(frame_count1+endpt)):
        r_handpts_list[i][j*3] = xnew_rev[idx]
        r_handpts_list[i][j*3+1] = ynew_rev[idx]
        r_handpts_list[i][j*3+2] = 0.5
        #r_handpts_list_new.append([xnew_rev[idx], ynew_rev[idx], 0.5])
        idx = idx+1
        
for j in range(21):
    xrev_list = []
    yrev_list = []
    trev_list = []
    
    skp_this = False

    for i in range(startpt-10, startpt):
        if l_handpts_list[i][j*3] <= 0 or l_handpts_list[i][j*3 + 1] <= 0:
            skp_this = True
            continue
        trev_list.append(i)
        xrev_list.append(l_handpts_list[i][j*3])
        yrev_list.append(l_handpts_list[i][j*3 + 1])
        # if posepts_list[i][j*3 + 1] >= _yymax:
        #     yrev_list.append(_yymax)
        # else:
        #     yrev_list.append(l_handpts_list[i][j*3 + 1])
        
    for i in range(endpt+int(frame_count1), endpt+int(frame_count1)+10):
        if l_handpts_list[i][j*3] <= 0 or l_handpts_list[i][j*3 + 1] <= 0:
            skp_this = True
            continue
        trev_list.append(i)
        xrev_list.append(l_handpts_list[i][j*3])
        yrev_list.append(l_handpts_list[i][j*3 + 1])
        # if posepts_list[i][j*3 + 1] >= _yymax:
        #     yrev_list.append(_yymax)
        # else:
        #     yrev_list.append(l_handpts_list[i][j*3 + 1])
        
    if skp_this == True:
        skp_this = False
        continue
    
    zx_rev = np.polyfit(np.array(trev_list), np.array(xrev_list), 3)
    px_rev = np.poly1d(zx_rev)
    zy_rev = np.polyfit(np.array(trev_list), np.array(yrev_list), 3)
    py_rev = np.poly1d(zy_rev)
    xnew_rev = px_rev(timspace)
    ynew_rev = py_rev(timspace) 
    
    idx = 0
    for i in range(startpt, int(frame_count1+endpt)):
        l_handpts_list[i][j*3] = xnew_rev[idx]
        l_handpts_list[i][j*3+1] = ynew_rev[idx]
        l_handpts_list[i][j*3+2] = 0.5
        #l_handpts_list_new.append([xnew_rev[idx], ynew_rev[idx], 0.5])
        idx = idx+1

        
# for j in range(70):
#     xrev_list = []
#     yrev_list = []
#     trev_list = []

#     for i in range(startpt-10, startpt):
#         trev_list.append(i)
#         xrev_list.append(facepts_list[i][j*3])
#         yrev_list.append(facepts_list[i][j*3+1])
        
#     for i in range(endpt+int(frame_count1), endpt+int(frame_count1)+10):
#         trev_list.append(i)
#         xrev_list.append(facepts_list[i][j*3])
#         yrev_list.append(facepts_list[i][j*3+1])
        
#     zx_rev = np.polyfit(np.array(trev_list), np.array(xrev_list), 1)
#     px_rev = np.poly1d(zx_rev)
#     zy_rev = np.polyfit(np.array(trev_list), np.array(yrev_list), 1)
#     py_rev = np.poly1d(zy_rev)
#     xnew_rev = px_rev(timspace)
#     ynew_rev = py_rev(timspace) 
    
#     idx = 0
#     for i in range(startpt, int(frame_count1+endpt)):
#         facepts_list[i][j*3] = xnew_rev[idx]
#         facepts_list[i][j*3+1] = ynew_rev[idx]
#         facepts_list[i][j*3+2] = 0.5
#         #facepts_list_new.append([xnew_rev[idx], ynew_rev[idx], 0.5])
#         idx = idx+1

skip_idx = 0
skip_frames = 4#int(avg_fps/2)
for i in range(len(posepts_list)):
    
    if i > startpt and i < int(frame_count1+endpt):
        #continue
        if skip_idx != skip_frames:
            skip_idx = skip_idx + 1
            continue
        else:
            skip_idx = 0
            #print(posepts_list[i])
    
    output_frame = np.zeros((512, 1024, 3), np.uint8)
    output_frame.fill(255)
    hand_frame = output_frame.copy()

    posepts_list_new.append(posepts_list[i])
    facepts_list_new.append(facepts_list[i])
    r_handpts_list_new.append(r_handpts_list[i])
    l_handpts_list_new.append(l_handpts_list[i])
    display_skleton(output_frame, posepts_list[i], facepts_list[i], r_handpts_list[i], l_handpts_list[i], zeropts=[0,0])
    
    plt.plot(i, posepts_list[i][7*3], 'g*')
    
    #display_single_keypoints_list(hand_frame, r_handpts_list[i])
    
    #cv2.putText(output_frame, str(i), (30,30), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (255, 0, 0) , 1, cv2.LINE_AA) 
    
    if args.save_dir:
        _fn = "frame_"+'{:0>12}'.format(i)+".png"
        _filepath_label = os.path.join(args.save_dir, "test_label")
        filename_label = os.path.join(_filepath_label, _fn)
        cv2.imwrite(filename_label, output_frame)
    cv2.imshow("Skleton Frame", output_frame)
    #cv2.imshow("Hand Frame", hand_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
#plt.show()
keypoints_prev = {'posepts' : posepts_list_new, 'facepts':facepts_list_new, 'r_handpts':r_handpts_list_new, 'l_handpts':l_handpts_list_new, 'fps': avg_fps}
if args.out_pkl:
    import pickle
    outfile = open(args.out_pkl, 'wb')
    pickle.dump(keypoints_prev, outfile)
    outfile.close()