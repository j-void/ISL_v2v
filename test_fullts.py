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
import cv2
import util.hand_utils as hand_utils

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

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    hand_bbox = [0, 0, 0, 0]
    prev_hand_bbox = [0, 0, 0, 0]
    bbox_size = data["max_bbox"]

    if opt.shand_gen:
      real_img = util.tensor2im(data['image'].data[0])
      lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True)
      lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
      rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
      #lbx, lby, lbw, rbx, rby, rbw = data['hand_bbox']
      lsx = (lbx+lbx+lbw)/2 - bbox_size/2
      lsx = 0 if lsx < 0 else lsx
      lsy = (lby+lby+lbw)/2 - bbox_size/2
      lsy = 0 if lsy < 0 else lsy
      rsx = (rbx+rbx+rbw)/2 - bbox_size/2
      rsx = 0 if rsx < 0 else rsx
      rsy = (rby+rby+rbw)/2 - bbox_size/2
      rsy = 0 if rsy < 0 else rsy
      hand_bbox = [lsx, lsy, rsx, rsy]

    if unset: #no previous results, condition on zero image
      previous_cond = torch.zeros(data['label'].size())
      prev_hand_bbox = hand_bbox
      unset = False    

  
    #generated = model.inference(data['label'], previous_cond, data['face_coords'])
    generated = model.inference(data['label'], previous_cond, hand_bbox, prev_hand_bbox, bbox_size)

    previous_cond = generated.data
    prev_hand_bbox = hand_bbox
    visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

time_taken = time.time() - initTime
print(f"Summary -> Total frames: {len(data_loader)}, Total time taken: {time_taken}s, Rate: {len(data_loader)/time_taken} FPS")

