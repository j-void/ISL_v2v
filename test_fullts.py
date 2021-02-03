### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_fullts
import util.util as util
from util.visualizer import Visualizer
from util import html
import numpy as np
import torch
import time
import util.hand_utils as hand_utils
import cv2
import glob

initTime = time.time()

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model_fullts(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
unset = True
print('#testing images = %d' % len(data_loader))

if not os.path.exists("test_tmp"):
    os.makedirs("test_tmp")

img_path = os.path.join(opt.dataroot, "test_img", "*.png")
imgs = glob.glob(img_path)
imgs.sort()

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    if unset: #no previous results, condition on zero image
      previous_cond = torch.zeros(data['label'].size())
      unset = False

    #targets = torch.cat((data['image'], data['next_image']), dim=3)
    real_img = cv2.cv2.imread(imgs[i]) #util.tensor2im(targets[0])
    height, width, channels = real_img.shape
    #real_img = cv2.cvtColor(real_img[:,:int(width/2),:], cv2.COLOR_RGB2BGR)
    hsk_frame = np.zeros(real_img.shape, dtype=np.uint8)
    hsk_frame.fill(255)
    
    if opt.netG == "global":
        scale_n, translate_n = hand_utils.resize_scale(real_img, myshape=(256, 512, 3))
        real_img = hand_utils.fix_image(scale_n, translate_n, real_img, myshape=(256, 512, 3))
        lhpts_real, rhpts_real, hand_state_real, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True, sz=64)
        hand_utils.display_single_hand_skleton(hsk_frame, lfpts)
        hand_utils.display_single_hand_skleton(hsk_frame, rfpts)
    else:
        scale_n, translate_n = hand_utils.resize_scale(real_img)
        real_img = hand_utils.fix_image(scale_n, translate_n, real_img)
        lhpts_real, rhpts_real, hand_state_real, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True)
        hand_utils.display_single_hand_skleton(hsk_frame, lfpts)
        hand_utils.display_single_hand_skleton(hsk_frame, rfpts)
        
    cv2.imwrite(os.path.join("test_tmp", "output_image_"+str(i)+"_"+'{:0>12}'.format(i)+".png"), hsk_frame)
    #generated = model.inference(data['label'], previous_cond, data['face_coords'])
    generated = model.inference(data['label'], previous_cond, hsk_frame)

    previous_cond = generated.data

    visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

time_taken = time.time() - initTime
print(f"Summary -> Total frames: {len(data_loader)}, Total time taken: {time_taken}s, Rate: {len(data_loader)/time_taken} FPS")

