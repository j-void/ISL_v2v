import cv2
import json
import numpy as np
import math

pose_colors = [[255,     0,    85], \
		[255,     0,     0], \
		[255,    85,     0], \
		[255,   170,     0], \
		[255,   255,     0], \
		[170,   255,     0], \
		[85,   255,     0], \
		[0,   255,     0], \
		[255,     0,     0], \
		[0,   255,    85], \
		[0,   255,   170], \
		[0,   255,   255], \
		[0,   170,   255], \
		[0,    85,   255], \
		[0,     0,   255], \
		[255,     0,   170], \
		[170,     0,   255], \
		[255,     0,   255], \
		[85,     0,   255], \
		[0,     0,   255], \
		[0,     0,   255], \
		[0,     0,   255], \
		[0,   255,   255], \
		[0,   255,   255], \
		[0,   255,   255]]
    
hand_colors = [[230, 53, 40], [231,115,64], [233, 136, 31], [213,160,13],[217, 200, 19], \
    [170, 210, 35], [139, 228, 48], [83, 214, 45], [77, 192, 46], \
    [83, 213, 133], [82, 223, 190], [80, 184, 197], [78, 140, 189], \
    [86, 112, 208], [83, 73, 217], [123,46,183], [189, 102,255], \
    [218, 83, 232], [229, 65, 189], [236, 61, 141], [255, 102, 145]]

faceSeq = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11], [11,12], [12,13], [13,14], [14,15], [15,16], \
    [17,18], [18,19], [19,20], [20,21], [22,23], [23,24], [24,25], [25,26], \
    [27,28], [28,29], [29,30], [31,32], [32,33], [33,34], [34,35], \
    [36,37], [37,38], [38,39], [39,40], [40,41], [41,36], [42,43], [43,44], [44,45], [45,46], [46,47], [47,42], \
    [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], [55,56], [56,57], [57,58], [58,59], [59,48], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,60]]

def readkeypointsfile_json(myfile):
	import json
	f = open(myfile, 'r')
	json_dict = json.load(f)
	people = json_dict['people']
	posepts =[]
	facepts = []
	r_handpts = []
	l_handpts = []
	for p in people:
		posepts += p['pose_keypoints_2d']
		facepts += p['face_keypoints_2d']
		r_handpts += p['hand_right_keypoints_2d']
		l_handpts += p['hand_left_keypoints_2d']

	return posepts, facepts, r_handpts, l_handpts

def display_keypoints(frame, posepts, facepts, r_handpts, l_handpts):
    for p in range(0, int(len(posepts)/3)):
        cv2.circle(frame, (int(posepts[p*3]), int(posepts[p*3+1])), 4, pose_colors[p], -1)
    for p in range(0, int(len(facepts)/3)):
        cv2.circle(frame, (int(facepts[p*3]), int(facepts[p*3+1])), 2, (255, 255, 255), -1)
    for p in range(0, int(len(r_handpts)/3)):
        cv2.circle(frame, (int(r_handpts[p*3]), int(r_handpts[p*3+1])), 4, hand_colors[p], -1)
    for p in range(0, int(len(l_handpts)/3)):
        cv2.circle(frame, (int(l_handpts[p*3]), int(l_handpts[p*3+1])), 4, hand_colors[p], -1)

def display_skleton(frame, posepts, facepts, r_handpts, l_handpts):
    
    
    # limbSeq = [[0,1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], \
	# 		[9, 10], [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 19], [14, 21], [15, 17], [16, 18], \
	# 		[19, 20], [22, 23]]
    
    limbSeq = [[0,1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [0, 15], [0, 16], [15, 17], [16, 18]]#, \
			# [9, 10], [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 19], [14, 21], \
			# [19, 20], [22, 23]]
    
    handSeq = [[0,1], [1,2], [2,3], [3,4], [0,5], [5,6], [6,7], [7,8], [0,9], [9,10], [10,11], [11,12], [0,13], [13,14], [14,15], [15,16], [0,17], [17,18], [18,19], [19,20]]
    
    posepts_2d = []
    r_handpts_2d = np.zeros((21, 3))
    l_handpts_2d = np.zeros((21, 3))
    facepts_2d  = np.zeros((70, 2), dtype=np.int)
    
    for p in range(0, int(len(posepts)/3)):
        pt = (int(posepts[p*3]), int(posepts[p*3+1]))
        if (p <= 7) or (p >= 15 and p <= 18):
            # if (posepts[p*3+2] == 0):
            #     return False
            cv2.circle(frame, (int(posepts[p*3]), int(posepts[p*3+1])), 6, pose_colors[p], -1)
        posepts_2d.append(pt)
        
    for p in range(21):
        # if (r_handpts[p*3+2] < 0.1):
        #     return False
        pt = (int(r_handpts[p*3]), int(r_handpts[p*3+1]))
        r_handpts_2d[p, 0] = int(r_handpts[p*3])
        r_handpts_2d[p, 1] = int(r_handpts[p*3+1])
        r_handpts_2d[p, 2] = r_handpts[p*3+2]
        if r_handpts[p*3] > 0 and r_handpts[p*3+1] > 0 and r_handpts[p*3+2] > 0.3:
            cv2.circle(frame, (int(r_handpts[p*3]), int(r_handpts[p*3+1])), 4, hand_colors[p], -1)
        #r_handpts_2d.append(pt)
        
    # if np.sum(r_handpts_2d[:,2])/21 < 0.2 and len(r_handpts) > 0:
    #     return False
        
    for p in range(21):
        # if (l_handpts[p*3+2] < 0.1):
        #     return False
        pt = (int(l_handpts[p*3]), int(l_handpts[p*3+1]))
        l_handpts_2d[p, 0] = int(l_handpts[p*3])
        l_handpts_2d[p, 1] = int(l_handpts[p*3+1])
        l_handpts_2d[p, 2] = l_handpts[p*3+2]
        if l_handpts[p*3] > 0 and l_handpts[p*3+1] > 0 and l_handpts[p*3+2] > 0.3:
            cv2.circle(frame, (int(l_handpts[p*3]), int(l_handpts[p*3+1])), 4, hand_colors[p], -1)
        #l_handpts_2d.append(pt)
        
    if np.sum(l_handpts_2d[:,2])/21 < 0.2 and np.sum(r_handpts_2d[:,2])/21 < 0.2 and (len(l_handpts)>0 or len(r_handpts) > 0):
        return False
        
    for p in range(0, int(len(facepts)/3)):
        pt = (int(facepts[p*3]), int(facepts[p*3+1]))
        if facepts[p*3] > 0 and facepts[p*3+1] > 0 and facepts[p*3+2] > 0.3:
            cv2.circle(frame, (int(facepts[p*3]), int(facepts[p*3+1])), 1, (0, 0, 0), -1)
            facepts_2d[p, 0] = int(facepts[p*3])
            facepts_2d[p, 1] = int(facepts[p*3+1])
    
    for k in range(len(limbSeq)):
        firstlimb_ind = limbSeq[k][0]
        secondlimb_ind = limbSeq[k][1]
        if posepts_2d[firstlimb_ind][0] > 0 and posepts_2d[secondlimb_ind][0] > 0:
            cv2.line(frame, posepts_2d[firstlimb_ind], posepts_2d[secondlimb_ind], pose_colors[k], 5)
    
    
    for k in range(len(faceSeq)):
        firstlimb_ind = faceSeq[k][0]
        secondlimb_ind = faceSeq[k][1]
        fp = (facepts_2d[firstlimb_ind, 0], facepts_2d[firstlimb_ind, 1])
        sp = (facepts_2d[secondlimb_ind, 0], facepts_2d[secondlimb_ind, 1])
        if fp[0] > 0 and sp[0] > 0 and fp[1] > 0 and sp[1] > 0:
            cv2.line(frame, fp, sp, (0,0,0), 1)
            
            
    for k in range(len(handSeq)):
        firstlimb_ind = handSeq[k][0]
        secondlimb_ind = handSeq[k][1]
        if r_handpts_2d[firstlimb_ind, 2] > 0.3 and r_handpts_2d[secondlimb_ind, 2] > 0.3:
            cv2.line(frame, (int(r_handpts_2d[firstlimb_ind, 0]), int(r_handpts_2d[firstlimb_ind, 1])), (int(r_handpts_2d[secondlimb_ind, 0]), int(r_handpts_2d[secondlimb_ind, 1])), hand_colors[k], 4)
        if l_handpts_2d[firstlimb_ind, 2] > 0.3 and l_handpts_2d[secondlimb_ind, 2] > 0.3:
            cv2.line(frame, (int(l_handpts_2d[firstlimb_ind, 0]), int(l_handpts_2d[firstlimb_ind, 1])), (int(l_handpts_2d[secondlimb_ind, 0]), int(l_handpts_2d[secondlimb_ind, 1])), hand_colors[k], 4)
            
    return True
            
def resize_scale(frame, myshape = (512, 1024, 3)):
    curshape = frame.shape
    if curshape == myshape:
        scale = 1
        translate = (0.0, 0.0)
        return scale, translate

    x_mult = myshape[0] / float(curshape[0])
    y_mult = myshape[1] / float(curshape[1])

    if x_mult == y_mult:
        scale = x_mult
        translate = (0.0, 0.0)
    elif y_mult > x_mult:
        y_new = x_mult * float(curshape[1])
        translate_y = (myshape[1] - y_new) / 2.0
        scale = x_mult
        translate = (translate_y, 0.0)
    elif x_mult > y_mult:
        x_new = y_mult * float(curshape[0])
        translate_x = (myshape[0] - x_new) / 2.0
        scale = y_mult
        translate = (0.0, translate_x)
        
    # M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    # output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return scale, translate

def fix_image(scale, translate, frame, myshape = (512, 1024, 3)):
    M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return output_image

def fix_scale_coords(points, scale, translate):
    points = np.array(points)
    points[0::3] = scale * points[0::3] + translate[0]
    points[1::3] = scale * points[1::3] + translate[1]
    return list(points)

iHSeq_right = [[20,3], [3,2], [2,1], [1,0], \
    [20,7], [7,6], [6,5], [5,4], \
    [20,9], [9,8], [8,7], [7,6], \
    [20,15], [15,14], [14,13], [13,12], \
    [20,19], [19,18], [18,17], [17,16]]

iHSeq_left = [[21,22], [22,23], [23,24], [24,41], \
    [25,26], [26,27], [27,28], [28,41], \
    [29,30], [30,31], [31,32], [32,41], \
    [33,34], [34,35], [35,36], [36,41], \
    [37,38], [38,39], [39,40], [40,41]]

def display_single_hand_skleton_right(frame, handpts):
                        
    for k in range(len(iHSeq_right)):
        firstlimb_ind = iHSeq_right[k][0]
        secondlimb_ind = iHSeq_right[k][1]
        cv2.line(frame, (int(handpts[firstlimb_ind, 0]), int(handpts[firstlimb_ind, 1])), (int(handpts[secondlimb_ind, 0]), int(handpts[secondlimb_ind, 1])), (255,255,255), 2)

    for p in range(21):
        cv2.circle(frame, (int(handpts[p,0]), int(handpts[p,1])), 4, hand_colors[p], -1)
        #frame = cv2.putText(frame, str(p), (int(handpts[p,0]), int(handpts[p,1])), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (255, 0, 0) , 1, cv2.LINE_AA) 
            
    return True

def get_keypoint_array(pts):
    pts_arry = np.zeros((21,2))
    prob = 0
    for p in range(21):
        pts_arry[p,0] = pts[p*3]
        pts_arry[p,1] = pts[p*3+1]
        prob = prob + pts[p*3+2]
    
    return pts_arry

def display_single_hand_skleton_left(frame, handpts):
                        
    for k in range(len(iHSeq_left)):
        firstlimb_ind = iHSeq_left[k][1]
        secondlimb_ind = iHSeq_left[k][0]
        cv2.line(frame, (int(handpts[firstlimb_ind, 0]), int(handpts[firstlimb_ind, 1])), (int(handpts[secondlimb_ind, 0]), int(handpts[secondlimb_ind, 1])), (255,255,255), 2)

    for p in range(21, 42):
        cv2.circle(frame, (int(handpts[p,0]), int(handpts[p,1])), 4, hand_colors[p-21], -1)
            
    return True

