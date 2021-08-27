### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from re import L
import time
from collections import OrderedDict

from numpy.lib import utils
from numpy.lib.ufunclike import fix
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_fullts, create_model_refine
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import util.hand_utils as hand_utils
import cv2

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

""" new residual model """
model = create_model_fullts(opt)
model_refine = create_model_refine(opt)
visualizer = Visualizer(opt)

tmp_out_path = os.path.join(opt.checkpoints_dir, opt.name, "tmp")

unset = True

with open(os.path.join(opt.dataroot, "bbox_size.txt"), 'r') as f:
    bbox_size = int(f.read())

if not os.path.exists(tmp_out_path):
    os.makedirs(tmp_out_path)


total_steps = (start_epoch-1) * dataset_size + epoch_iter    
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0
        
        ############## Forward Pass ######################
        
        if opt.shand_gen:
            real_img = util.tensor2im(data['image'].data[0])
            real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
            lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True)
            lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
            rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
            lsx = (lbx+lbx+lbw)/2 - bbox_size/2
            lsx = 0 if lsx < 0 else int(lsx)
            lsy = (lby+lby+lbw)/2 - bbox_size/2
            lsy = 0 if lsy < 0 else int(lsy)
            rsx = (rbx+rbx+rbw)/2 - bbox_size/2
            rsx = 0 if rsx < 0 else int(rsx)
            rsy = (rby+rby+rbw)/2 - bbox_size/2
            rsy = 0 if rsy < 0 else int(rsy)
            hand_bbox = [lsx, lsy, rsx, rsy, lbw, rbw]
        
        if unset: #no previous results, condition on zero image
            previous_cond = torch.zeros(data['label'].size())
            unset = False 
        
        generated = model.inference(data['label'], previous_cond, hand_bbox, bbox_size)
        previous_cond = generated.data
        
        cond_zeros = torch.zeros(data['image'].size()).float()
        bbox_size_ = 0
        hand_bbox_ = [0, 0, 0, 0, 0, 0]
        if opt.refine_hand:
            bbox_size_ = bbox_size + 20
            lsx = (lbx+lbx+lbw)/2 - bbox_size_/2
            lsx = 0 if lsx < 0 else int(lsx)
            lsy = (lby+lby+lbw)/2 - bbox_size_/2
            lsy = 0 if lsy < 0 else int(lsy)
            rsx = (rbx+rbx+rbw)/2 - bbox_size_/2
            rsx = 0 if rsx < 0 else int(rsx)
            rsy = (rby+rby+rbw)/2 - bbox_size_/2
            rsy = 0 if rsy < 0 else int(rsy)
            hand_bbox_ = [lsx, lsy, rsx, rsy, lbw, rbw]
        
        losses, generated_refine = model_refine(Variable(data['image']), Variable(generated.data), Variable(cond_zeros), hand_bbox_, bbox_size_, infer=True)
        # losses, generated_refine = model_refine(Variable(data['label']), Variable(data['next_label']), Variable(data['image']), \
        #             Variable(data['next_image']), Variable(cond_zeros), hand_bbox, bbox_size, infer=True)
        #model_refine(Variable(data['image']), Variable(cond_zeros), infer=True)
        
        #output_image = cv2.hconcat([cv2.cvtColor(util.tensor2im(generated_refine[0].data[0]), cv2.COLOR_RGB2BGR), cv2.cvtColor(util.tensor2im(generated.data[0]), cv2.COLOR_RGB2BGR), real_img])
        #cv2.imwrite("output_image.png", output_image)
        
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model_refine.module.loss_names, losses))
        
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + (loss_dict['D_hand_left_real'] + loss_dict['D_hand_left_fake']) * 0.5 + (loss_dict['D_hand_right_real'] + loss_dict['D_hand_right_fake']) * 0.5 
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_GAN_hand_left'] + loss_dict['G_GAN_hand_right']
        
        ############### Backward Pass ####################
        # update generator weights
        model_refine.module.optimizer_G.zero_grad()
        loss_G.backward()
        model_refine.module.optimizer_G.step()

        # update discriminator weights
        model_refine.module.optimizer_D.zero_grad()
        loss_D.backward()
        model_refine.module.optimizer_D.step()
        
        if total_steps % opt.print_freq == 0:
            errors = {}
            if torch.__version__[0] == '1':
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            else:
                errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
        
        ### display output images            
        if total_steps % opt.save_latest_freq == 0:            
            output_image = cv2.hconcat([cv2.cvtColor(util.tensor2im(generated_refine[0].data[0]), cv2.COLOR_RGB2BGR), cv2.cvtColor(util.tensor2im(generated.data[0]), cv2.COLOR_RGB2BGR), real_img])
            #output_image = cv2.hconcat([cv2.cvtColor(util.tensor2im(generated_refine[0].data[0]), cv2.COLOR_RGB2BGR), real_img])
            cv2.imwrite(os.path.join(tmp_out_path, "output_image_"+str(epoch)+"_"+'{:0>12}'.format(i)+".png"), output_image)
            if opt.refine_hand:
                cv2.imwrite(os.path.join(tmp_out_path, "hand_left"+str(epoch)+"_"+'{:0>12}'.format(i)+".png"), cv2.cvtColor(util.tensor2im(generated_refine[1].data[0]), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(tmp_out_path, "hand_right"+str(epoch)+"_"+'{:0>12}'.format(i)+".png"), cv2.cvtColor(util.tensor2im(generated_refine[2].data[0]), cv2.COLOR_RGB2BGR))
            

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')
            model_refine.module.save('latest')      
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
    # end of epoch  
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.save('latest')
        model.save(epoch)
        model_refine.module.save('latest')
        model_refine.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        print('------------- finetuning Local + Global generators jointly -------------')
        model_refine.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model_refine.module.update_learning_rate()
