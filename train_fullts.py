### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from re import L
import time
from collections import OrderedDict

from numpy.lib import utils
from numpy.lib.ufunclike import fix
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_fullts
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
visualizer = Visualizer(opt)

tmp_out_path = os.path.join(opt.checkpoints_dir, opt.name, "tmp")

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


        no_nexts = data['next_label'].dim() > 1 #check if has a next label (last training pair does not have a next label)

        if no_nexts:
            cond_zeros = torch.zeros(data['label'].size()).float()
            
            hand_bbox = [0, 0, 0, 0, 0, 0]
            next_hand_bbox = [0, 0, 0, 0, 0, 0]
            
            
            if opt.shand_gen:
                real_img = util.tensor2im(data['image'].data[0])
                height, width, channels = real_img.shape
                lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True)
                lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
                rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
                
                #lbx, lby, lbw, rbx, rby, rbw = data['hand_bbox']
                
                lsx = (lbx+lbx+lbw)/2 - bbox_size/2
                lsx = 0 if lsx < 0 else int(lsx)
                lsy = (lby+lby+lbw)/2 - bbox_size/2
                lsy = 0 if lsy < 0 else int(lsy)
                rsx = (rbx+rbx+rbw)/2 - bbox_size/2
                rsx = 0 if rsx < 0 else int(rsx)
                rsy = (rby+rby+rbw)/2 - bbox_size/2
                rsy = 0 if rsy < 0 else int(rsy)
                hand_bbox = [lsx, lsy, rsx, rsy, lbw, rbw]
                print("hand_bbox", hand_bbox, lbx, lby, lbw, rbx, rby, rbw)
                
                next_img = util.tensor2im(data['next_image'].data[0])
                lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(next_img, fix_coords=True)
                lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
                rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
                #lbx, lby, lbw, rbx, rby, rbw = data['next_hand_bbox']
                lsx = (lbx+lbx+lbw)/2 - bbox_size/2
                lsx = 0 if lsx < 0 else int(lsx)
                lsy = (lby+lby+lbw)/2 - bbox_size/2
                lsy = 0 if lsy < 0 else int(lsy)
                rsx = (rbx+rbx+rbw)/2 - bbox_size/2
                rsx = 0 if rsx < 0 else int(rsx)
                rsy = (rby+rby+rbw)/2 - bbox_size/2
                rsy = 0 if rsy < 0 else int(rsy)
                next_hand_bbox = [lsx, lsy, rsx, rsy, lbw, rbw]
                print("next_hand_bbox", next_hand_bbox, lbx, lby, lbw, rbx, rby, rbw )
            
            losses, generated = model(Variable(data['label']), Variable(data['next_label']), Variable(data['image']), \
                    Variable(data['next_image']), Variable(cond_zeros), hand_bbox, next_hand_bbox, bbox_size, infer=True)


            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + (loss_dict['D_hand_left_real'] + loss_dict['D_hand_left_fake']) * 0.5 + (loss_dict['D_hand_right_real'] + loss_dict['D_hand_right_fake']) * 0.5 
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_GAN_hand_left'] + loss_dict['G_GAN_hand_right']

            ############### Backward Pass ####################
            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
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
                syn_img_hand = util.tensor2im(generated[0].data[0])
                height, width, channels = syn_img_hand.shape
                syn_img_hand = cv2.cvtColor(syn_img_hand[:,int(width/2):,:], cv2.COLOR_RGB2BGR)
                real_hand_img = cv2.cvtColor(util.tensor2im(data['image'].data[0]), cv2.COLOR_RGB2BGR)
                inputs = torch.cat((data['label'], data['next_label']), dim=3)
                input_label = util.tensor2im(inputs[0])[:,int(width/2):,:]
                input_label = cv2.cvtColor(input_label, cv2.COLOR_RGB2BGR)
                if opt.netG == "global":
                    scale_n, translate_n = hand_utils.resize_scale(input_label, myshape=(256, 512, 3))
                    input_label = hand_utils.fix_image(scale_n, translate_n, input_label, myshape=(256, 512, 3))
                else:
                    scale_n, translate_n = hand_utils.resize_scale(input_label)
                    input_label = hand_utils.fix_image(scale_n, translate_n, input_label)
                
                output_image = cv2.hconcat([syn_img_hand, real_hand_img, input_label])
                
                if opt.shand_gen:
                    cv2.imwrite(os.path.join(tmp_out_path, "output_hand_left_"+str(epoch)+"_"+'{:0>12}'.format(i)+".png"), cv2.cvtColor(util.tensor2im(generated[4].data[0]), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(tmp_out_path, "output_hand_right_"+str(epoch)+"_"+'{:0>12}'.format(i)+".png"), cv2.cvtColor(util.tensor2im(generated[5].data[0]), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(tmp_out_path, "output_image_"+str(epoch)+"_"+'{:0>12}'.format(i)+".png"), output_image)
            
            # if save_fake and opt.hand_discrim:
            #     syn = generated[0].data[0]
            #     inputs = torch.cat((data['label'], data['next_label']), dim=3)
            #     targets = torch.cat((data['image'], data['next_image']), dim=3)
            #     syn_img_hand = util.tensor2im(syn)[:,:1024,:]
            #     syn_img_hand = cv2.cvtColor(syn_img_hand, cv2.COLOR_RGB2BGR)
            #     lhpts_gen, rhpts_gen = hand_utils.get_keypoints(syn_img_hand)
            #     lhpts_gen = hand_utils.rescale_points(1024, 512, lhpts_gen)
            #     rhpts_gen = hand_utils.rescale_points(1024, 512, rhpts_gen)
            #     hand_utils.display_hand_skleton(syn_img_hand, lhpts_gen, rhpts_gen)
            #     syn_img_hand = cv2.cvtColor(syn_img_hand, cv2.COLOR_BGR2RGB)
            #     real_hand_img = real_img.copy()
            #     lhpts_real_r = hand_utils.rescale_points(1024, 512, lhpts_real)
            #     rhpts_real_r = hand_utils.rescale_points(1024, 512, rhpts_real)
            #     hand_utils.display_hand_skleton(real_hand_img, lhpts_real_r, rhpts_real_r)
            #     real_hand_img = cv2.cvtColor(real_hand_img, cv2.COLOR_BGR2RGB)
            #     visuals = OrderedDict([('input_label', util.tensor2im(inputs[0], normalize=False)),
            #                                ('synthesized_image', util.tensor2im(syn)),
            #                                ('real_image', util.tensor2im(targets[0]),
            #                                 ('syn_hand_image', syn_img_hand),
            #                                 ('real_hand_image', real_hand_img))])
            #     # if opt.face_generator: #display face generator on tensorboard
            #     #     miny, maxy, minx, maxx = data['face_coords'][0]
            #     #     res_face = generated[2].data[0]
            #     #     syn_face = generated[1].data[0]
            #     #     preres = generated[3].data[0]
            #     #     visuals = OrderedDict([('input_label', util.tensor2im(inputs[0], normalize=False)),
            #     #                            ('synthesized_image', util.tensor2im(syn)),
            #     #                            ('synthesized_face', util.tensor2im(syn_face)),
            #     #                            ('residual', util.tensor2im(res_face)),
            #     #                            ('real_face', util.tensor2im(data['image'][0][:, miny:maxy, minx:maxx])),
            #     #                            # ('pre_residual', util.tensor2im(preres)),
            #     #                            # ('pre_residual_face', util.tensor2im(preres[:, miny:maxy, minx:maxx])),
            #     #                            ('input_face', util.tensor2im(data['label'][0][:, miny:maxy, minx:maxx], normalize=False)),
            #     #                            ('real_image', util.tensor2im(targets[0]))])
            #     visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
    # end of epoch  
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        print('------------- finetuning Local + Global generators jointly -------------')
        model.module.update_fixed_params()

    ### instead of only training the face discriminator, train the entire network after certain iterations
    if (opt.niter_fix_main != 0) and (epoch == opt.niter_fix_main):
        print('------------- traing all the discriminators now and not just the face -------------')
        model.module.update_fixed_params_netD()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
