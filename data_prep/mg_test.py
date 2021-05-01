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
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
from lmfit import Model
from scipy.signal import savgol_filter
import rdp as rdp
import math

initTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--k1_pkl", type=str, help="word1 directory")
parser.add_argument("--k2_pkl", type=str, help='word2 directory')
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--train_dir", type=str, help='train directory')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

diff = 0.2
t1 = 0.2
t2 = 0.2

keypoints1 = joblib.load(args.k1_pkl)
keypoints2 = joblib.load(args.k2_pkl)

posepts_list = keypoints1["posepts"] + keypoints2["posepts"]
facepts_list = keypoints1["facepts"] + keypoints2["facepts"]
r_handpts_list = keypoints1["r_handpts"] + keypoints2["r_handpts"]
l_handpts_list = keypoints1["l_handpts"] + keypoints2["l_handpts"]


frame_count1 = len(keypoints1["posepts"])
frame_count2 = len(keypoints2["posepts"])

s1 = int(t1*frame_count1)
s2 = int(t2*frame_count2)



def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exponential(x, a, b):
    return a*np.exp(b*x)

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

def nearest(plot_list) :
    min_val = min(plot_list)
    new_list = plot_list.copy()
    new_list.sort()
    for val in new_list:
        if val > min_val:
            min_val = val
            break
    
    for i in range(len(plot_list)):
        if plot_list[i] < 0:
            plot_list[i] = min_val

xplot_list = []
yplot_list = []
time_seq = []

    
j=4
xplot_list = []
yplot_list = []
time_seq = []

timspace = np.arange(int(frame_count1-s1), int(frame_count1+s2))

all_ypts = []
all_xpts = []
for i in range(len(posepts_list)):
    posepts_arr = get_keypoint_array_pose(posepts_list[i])
    all_xpts.append(posepts_arr[j, 0])
    all_ypts.append(posepts_arr[j, 1])

plt.figure(1)
plt.plot(np.arange(len(posepts_list)), all_ypts, 'yo')
plt.title("Y Plot")
plt.figure(2)
plt.plot(np.arange(len(posepts_list)), all_xpts, 'yo')
plt.title("X Plot")

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

diff1 = 100000
for val in k1_ypts[::-1]:
    if val < 0:
        continue
    if (val - 416) < 0:
        pt = k1_ypts.index(val)
        startpt = pt
        plt.figure(1)
        plt.plot(pt, k1_ypts[pt],'k*')
        break
    
endpt = 0 
for val in k2_ypts:
    if val < 0:
        continue
    if (val - 416) < 0:
        pt = k2_ypts.index(val)
        endpt = pt
        plt.figure(1)
        plt.plot(pt+len(k1_ypts), k2_ypts[pt],'k*')
        break
                

timspace = np.arange(startpt, endpt+frame_count1)

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


# k1_yhat = savgol_filter(_k1_ypts, 11, 3)
# _yinit = np.argwhere(k1_ypts==_k1_ypts[0])[0][0]
# plt.figure(1)
# plt.plot(np.arange(_yinit, len(k1_yhat)+_yinit), k1_yhat, 'b-')

min_angle = np.pi*0.22


simplified_trajectory_k1y = np.array(rdp.rdp(_k1_ypts_list, epsilon=10))
st_k1y, sy_k1 = simplified_trajectory_k1y.T
directions_k1y = np.diff(simplified_trajectory_k1y, axis=0)
theta_k1y = rdp.angle(directions_k1y)
idx_k1y = np.where(theta_k1y>min_angle)[0]+1

if startpt == 0:
    startpt = np.where(k1_ypts==sy_k1[idx_k1y[-1]])[0][0]
    print(startpt)

plt.figure(1)
plt.plot(st_k1y, sy_k1, 'r-')
plt.plot(st_k1y[idx_k1y], sy_k1[idx_k1y], 'g*')

simplified_trajectory_k2y = np.array(rdp.rdp(_k2_ypts_list, epsilon=10))
st_k2y, sy_k2 = simplified_trajectory_k2y.T
directions_k2y = np.diff(simplified_trajectory_k2y, axis=0)
theta_k2y = rdp.angle(directions_k2y)
idx_k2y = np.where(theta_k2y>min_angle)[0]+1

plt.figure(1)
plt.plot(st_k2y, sy_k2, 'r-')
plt.plot(st_k2y[idx_k2y], sy_k2[idx_k2y], 'g*')



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


# k1_xhat = savgol_filter(_k1_xpts, 11, 3)
# _xinit = np.argwhere(k1_xpts==_k1_xpts[0])[0][0]
# plt.figure(1)
# plt.plot(np.arange(_xinit, len(k1_xhat)+_xinit), k1_xhat, 'b-')

min_angle = np.pi*0.22


simplified_trajectory_k1x = np.array(rdp.rdp(_k1_xpts_list, epsilon=5))
st_k1x, sx_k1 = simplified_trajectory_k1x.T
directions_k1x = np.diff(simplified_trajectory_k1x, axis=0)
theta_k1x = rdp.angle(directions_k1x)
idx_k1x = np.where(theta_k1x>min_angle)[0]+1



plt.figure(2)
plt.plot(st_k1x, sx_k1, 'r-')
plt.plot(st_k1x[idx_k1x], sx_k1[idx_k1x], 'g*')

simplified_trajectory_k2x = np.array(rdp.rdp(_k2_xpts_list, epsilon=5))
st_k2x, sx_k2 = simplified_trajectory_k2x.T
directions_k2x = np.diff(simplified_trajectory_k2x, axis=0)
theta_k2x = rdp.angle(directions_k2x)
idx_k2x = np.where(theta_k2x>min_angle)[0]+1

plt.figure(2)
plt.plot(st_k2x, sx_k2, 'r-')
plt.plot(st_k2x[idx_k2x], sx_k2[idx_k2x], 'g*')


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
        

plt.figure(3)
plt.title("Distance Plot")
plt.xlim([0, len(k1_ypts)+len(k2_xpts)])
plt.plot(k1_all_dist_t, k1_all_dist, 'yo')
plt.plot(k2_all_dist_t, k2_all_dist, 'yo')
# plt.plot(k2_z_dist, [0] * len(k2_z_dist), 'g*')
# plt.plot(k1_z_dist, [0] * len(k1_z_dist), 'g*')    

simplified_trajectory_k1d = np.array(rdp.rdp(_dist_k1, epsilon=5))
st_k1d, sd_k1 = simplified_trajectory_k1d.T
directions_k1d = np.diff(simplified_trajectory_k1d, axis=0)
theta_k1d = rdp.angle(directions_k1d)
idx_k1d = np.where(theta_k1d>min_angle)[0]+1

for i in range(len(idx_k1d), 0, -1):
    #print(np.where(k1_all_dist==sd_k1[i])[0][0], abs(sd_k1[i-1] - sd_k1[i]))
    if abs(sd_k1[i-1] - sd_k1[i]) < 3:
        _tmpIdx = np.where(_dist_k1==sd_k1[i-1])[0][0]
        if _tmpIdx < startpt:
            if int(st_k1d[i-1]) > 5 and len(idx_k1d)-i < 4:
                print("Considering 0 displacement value", _tmpIdx)
                plt.figure(3)
                plt.plot(st_k1d[i-1], sd_k1[i-1], 'ko')
                startpt = int(st_k1d[i-1])
        break

plt.figure(3)
plt.plot(st_k1d, sd_k1, 'r-')
plt.plot(st_k1d[idx_k1d], sd_k1[idx_k1d], 'g*')


simplified_trajectory_k2d = np.array(rdp.rdp(_dist_k2, epsilon=5))
st_k2d, sd_k2 = simplified_trajectory_k2d.T
directions_k2d = np.diff(simplified_trajectory_k2d, axis=0)
theta_k2d = rdp.angle(directions_k2d)
idx_k2d = np.where(theta_k2d>min_angle)[0]+1

plt.figure(3)
plt.plot(st_k2d, sd_k2, 'r-')
plt.plot(st_k2d[idx_k2d], sd_k2[idx_k2d], 'g*')

xrev_list = []
yrev_list = []
trev_list = []

#startpt = np.where(k1_ypts==sy_k1[idx_k1y[-2]])[0][0]
timspace = np.arange(startpt, endpt+frame_count1)

print(startpt, endpt+frame_count1)

for i in range(startpt-10, startpt):
    trev_list.append(i)
    posepts_arr = get_keypoint_array_pose(posepts_list[i])
    #plt.plot(i, posepts_arr[j, 1], 'b*')
    xrev_list.append(posepts_arr[j, 0])
    yrev_list.append(posepts_arr[j, 1])
    
for i in range(endpt+len(k1_ypts), endpt+len(k1_ypts)+10):
    trev_list.append(i)
    posepts_arr = get_keypoint_array_pose(posepts_list[i])
    #plt.plot(i, posepts_arr[j, 1], 'b*')
    xrev_list.append(posepts_arr[j, 0])
    yrev_list.append(posepts_arr[j, 1])
    
zx_rev = np.polyfit(np.array(trev_list), np.array(xrev_list), 1)
px_rev = np.poly1d(zx_rev)
zy_rev = np.polyfit(np.array(trev_list), np.array(yrev_list), 1)
py_rev = np.poly1d(zy_rev)
xnew_rev = px_rev(timspace)
ynew_rev = py_rev(timspace)  

plt.figure(1)
plt.plot(timspace, ynew_rev, 'b-')

x_new = np.linspace(posepts_list[int(frame_count1-s1)][j*3], posepts_list[int(frame_count1+s2)][j*3], (s1+s2))
y_new = np.linspace(posepts_list[int(frame_count1-s1)][j*3+1], posepts_list[int(frame_count1+s2)][j*3+1], (s1+s2))

#plt.plot(np.arange(int(frame_count1-s1), int(frame_count1+s2)), y_new, 'go')


plt.show()

# for i in range(len(posepts_list)):
#     output_frame = np.zeros((512, 1024, 3), np.uint8)
#     output_frame.fill(255)
#     display_skleton(output_frame, posepts_list[i], facepts_list[i], r_handpts_list[i], l_handpts_list[i])
#     cv2.putText(output_frame, str(i), (30,30), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (255, 0, 0) , 1, cv2.LINE_AA) 
#     cv2.imshow("Skleton Frame", output_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    