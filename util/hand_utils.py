import numpy as np
import mediapipe as mp
import cv2
from google.protobuf.json_format import MessageToDict

mp_hands = mp.solutions.hands

confidence = 0.7


hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=confidence)

hand_colors = [[230, 53, 40], [231,115,64], [233, 136, 31], [213,160,13],[217, 200, 19], \
    [170, 210, 35], [139, 228, 48], [83, 214, 45], [77, 192, 46], \
    [83, 213, 133], [82, 223, 190], [80, 184, 197], [78, 140, 189], \
    [86, 112, 208], [83, 73, 217], [123,46,183], [189, 102,255], \
    [218, 83, 232], [229, 65, 189], [236, 61, 141], [255, 102, 145]]

def get_keypoints(frame, fix_coords=False):
    lefthnd_pts = np.zeros((21, 2))
    righthnd_pts = np.zeros((21, 2))
    scale_n, translate_n = resize_scale(frame)
    image = fix_image(scale_n, translate_n, frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    hand_side = [None] * 2
    hand_prob = [0] * 2
    if not results.multi_hand_landmarks:
        #print("No multi_hand_landmarks")
        return lefthnd_pts, righthnd_pts, hand_prob
    for idx, hand_handedness in enumerate(results.multi_handedness):
        handedness_dict = MessageToDict(hand_handedness)
        hand_side[idx] = handedness_dict["classification"][0]["label"]
        if hand_side[idx] == "Left":
            hand_prob[0] = handedness_dict["classification"][0]["score"]
        elif hand_side[idx] == "Right":
            hand_prob[1] = handedness_dict["classification"][0]["score"]
    if hand_side[0] == hand_side[1]:
        if hand_prob[0] > hand_prob[1]:
            hand_side[1] = None
        else:
            hand_side[0] = None
    if results.multi_hand_landmarks:
        index = 0
        for hand_landmarks in results.multi_hand_landmarks:
            if hand_side[index] == "Left":
                if fix_coords:
                    lefthnd_pts = rescale_points(1024, 512, GetCoordForCurrentInstance(hand_landmarks))
                    x_start_l, y_start_l, box_size_l = assert_bbox(lefthnd_pts)
                    lefthnd_pts = restructure_points(lefthnd_pts, x_start_l, y_start_l)
                    lefthnd_pts = lefthnd_pts / box_size_l
                    lefthnd_pts = rescale_points(128, 128, lefthnd_pts)
                else:
                    lefthnd_pts = GetCoordForCurrentInstance(hand_landmarks)
            elif hand_side[index] == "Right":
                if fix_coords:
                    righthnd_pts = rescale_points(1024, 512, GetCoordForCurrentInstance(hand_landmarks))
                    x_start_r, y_start_r, box_size_r = assert_bbox(righthnd_pts)
                    righthnd_pts = restructure_points(righthnd_pts, x_start_r, y_start_r)
                    righthnd_pts = righthnd_pts / box_size_r
                    righthnd_pts = rescale_points(128, 128, righthnd_pts)
                else:
                    righthnd_pts = GetCoordForCurrentInstance(hand_landmarks)
            index = index + 1
    #print(lefthnd_pts, righthnd_pts)
    return lefthnd_pts, righthnd_pts, hand_prob

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic( static_image_mode=True ,min_detection_confidence=confidence)

def get_keypoints_holistic(frame, fix_coords=False):
    lefthnd_pts = np.zeros((21, 2))
    righthnd_pts = np.zeros((21, 2))
    scale_n, translate_n = resize_scale(frame)
    image = fix_image(scale_n, translate_n, frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    hand_state = [False, False]
    if results.left_hand_landmarks != None:
        hand_state[0] = True
        if fix_coords:
            lefthnd_pts = rescale_points(1024, 512, GetCoordForCurrentInstance(results.left_hand_landmarks))
            x_start_l, y_start_l, box_size_l = assert_bbox(lefthnd_pts)
            lefthnd_pts = restructure_points(lefthnd_pts, x_start_l, y_start_l)
            lefthnd_pts = lefthnd_pts / box_size_l
            lefthnd_pts = rescale_points(128, 128, lefthnd_pts)
        else:
            lefthnd_pts = GetCoordForCurrentInstance(results.left_hand_landmarks)
    
    if results.right_hand_landmarks != None:
        hand_state[1] = True
        if fix_coords:
            righthnd_pts = rescale_points(1024, 512, GetCoordForCurrentInstance(results.right_hand_landmarks))
            x_start_r, y_start_r, box_size_r = assert_bbox(righthnd_pts)
            righthnd_pts = restructure_points(righthnd_pts, x_start_r, y_start_r)
            righthnd_pts = righthnd_pts / box_size_r
            righthnd_pts = rescale_points(128, 128, righthnd_pts)
        else:
            righthnd_pts = GetCoordForCurrentInstance(results.right_hand_landmarks)
            
    return lefthnd_pts, righthnd_pts, hand_state
        
    
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
        
    return scale, translate

def fix_image(scale, translate, frame, myshape = (512, 1024, 3)):
    M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return output_image

def GetCoordForCurrentInstance(mp_output):
    hand_pts = np.zeros((21, 2))
    hand_pts[0, 0] = mp_output.landmark[mp_hands.HandLandmark.WRIST].x
    hand_pts[0, 1] = mp_output.landmark[mp_hands.HandLandmark.WRIST].y
    hand_pts[1, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_CMC].x
    hand_pts[1, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_CMC].y
    hand_pts[2, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_MCP].x
    hand_pts[2, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    hand_pts[3, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_IP].x
    hand_pts[3, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_IP].y
    hand_pts[4, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    hand_pts[4, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    hand_pts[5, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    hand_pts[5, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    hand_pts[6, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    hand_pts[6, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    hand_pts[7, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
    hand_pts[7, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    hand_pts[8, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    hand_pts[8, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    hand_pts[9, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    hand_pts[9, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    hand_pts[10, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
    hand_pts[10, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    hand_pts[11, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
    hand_pts[11, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    hand_pts[12, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    hand_pts[12, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    hand_pts[13, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
    hand_pts[13, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    hand_pts[14, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
    hand_pts[14, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    hand_pts[15, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
    hand_pts[15, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
    hand_pts[16, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
    hand_pts[16, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    hand_pts[17, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_MCP].x
    hand_pts[17, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_MCP].y
    hand_pts[18, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_PIP].x
    hand_pts[18, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    hand_pts[19, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_DIP].x
    hand_pts[19, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_DIP].y
    hand_pts[20, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    hand_pts[20, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    return hand_pts

def rescale_points(width, height, output):
    output[:,0] = output[:,0]*width
    output[:,1] = output[:,1]*height
    return output

def restructure_points(handpts, sx, sy):
    hpts = handpts
    hpts[:,0] = hpts[:,0] - sx
    hpts[:,1] = hpts[:,1] - sy 
    return hpts.astype(int)


def assert_bbox(handpts):
    x_mid = int(np.average(handpts[:,0]))
    y_mid = int(np.average(handpts[:,1]))
    x_min = int(np.min(handpts[:,0]))
    y_min = int(np.min(handpts[:,1]))
    x_max = int(np.max(handpts[:,0]))
    y_max = int(np.max(handpts[:,1]))
    
    max_dis = max(abs(x_max-x_min), abs(y_max-y_min))
    
    if x_mid - max_dis/2 > 0:
        sx = x_mid - max_dis/2 - max_dis*0.25
    else:
        sx = 0
    
    if y_mid - max_dis/2 > 0:
        sy = y_mid - max_dis/2 - max_dis*0.25
    else:
        sy = 0
    
    return int(sx), int(sy), int(max_dis*1.5)


handSeq = [[0,1], [1,2], [2,3], [3,4], \
    [0,5], [5,6], [6,7], [7,8], \
    [0,9], [9,10], [10,11], [11,12], \
    [0,13], [13,14], [14,15], [15,16], \
    [0,17], [17,18], [18,19], [19,20], \
    [5,9], [9,13], [13,17]]

def display_single_hand_skleton(frame, handpts):
                        
    for k in range(len(handSeq)):
        firstlimb_ind = handSeq[k][0]
        secondlimb_ind = handSeq[k][1]
        print(k)
        cv2.line(frame, (int(handpts[firstlimb_ind, 0]), int(handpts[firstlimb_ind, 1])), (int(handpts[secondlimb_ind, 0]), int(handpts[secondlimb_ind, 1])), hand_colors[k], 4)

    for p in range(handpts.shape[0]):
        cv2.circle(frame, (int(handpts[p,0]), int(handpts[p,1])), 4, (255, 0, 255), -1)
            
    return True

def display_hand_skleton(frame, r_handpts, l_handpts):
        
                
    for k in range(len(handSeq)):
        firstlimb_ind = handSeq[k][0]
        secondlimb_ind = handSeq[k][1]
        cv2.line(frame, (int(r_handpts[firstlimb_ind, 0]), int(r_handpts[firstlimb_ind, 1])), (int(r_handpts[secondlimb_ind, 0]), int(r_handpts[secondlimb_ind, 1])), (255,255,255), 4)
        cv2.line(frame, (int(l_handpts[firstlimb_ind, 0]), int(l_handpts[firstlimb_ind, 1])), (int(l_handpts[secondlimb_ind, 0]), int(l_handpts[secondlimb_ind, 1])), (255,255,255), 4)

    for p in range(r_handpts.shape[0]):
        cv2.circle(frame, (int(r_handpts[p,0]), int(r_handpts[p,1])), 4, (255, 0, 255), -1)
        cv2.circle(frame, (int(l_handpts[p,0]), int(l_handpts[p,1])), 4, (255, 0, 255), -1)   
            
    return True