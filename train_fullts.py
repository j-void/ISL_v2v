### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
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
            targets = torch.cat((data['image'], data['next_image']), dim=3)
            real_img = util.tensor2im(targets[0])
            height, width, channels = real_img.shape
            real_img = cv2.cvtColor(real_img[:,:int(width/2),:], cv2.COLOR_RGB2BGR)
            hsk_frame = np.zeros(real_img.shape, dtype=np.uint8)
            hsk_frame.fill(255)
            
            if opt.netG == "global":
                scale_n, translate_n = hand_utils.resize_scale(real_img, myshape=(256, 512, 3))
                real_img = hand_utils.fix_image(scale_n, translate_n, real_img, myshape=(256, 512, 3))
                lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True, sz=64)
                hand_utils.display_single_hand_skleton(hsk_frame, lfpts)
                hand_utils.display_single_hand_skleton(hsk_frame, rfpts)
                lx, ly, lw = hand_utils.get_mid(lfpts, int(bbox_size/2))
                rx, ry, rw = hand_utils.get_mid(rfpts, int(bbox_size/2))
                if lw != 0:
                    lh_label = hsk_frame[ly:ly+lw, lx:lx+lw, :]
                    lh_image = real_img[ly:ly+lw, lx:lx+lw, :]
                else:
                    lh_label = np.zeros((int(bbox_size/2), int(bbox_size/2), 3), dtype=np.uint8)
                    lh_label.fill(255)
                    lh_image = np.zeros((int(bbox_size/2), int(bbox_size/2), 3), dtype=np.uint8)
                    lh_image.fill(255)
                
                if rw != 0:
                    rh_label = hsk_frame[ry:ry+rw, rx:rx+rw, :]                
                    rh_image = real_img[ry:ry+rw, rx:rx+rw, :]
                else:
                    rh_label = np.zeros((int(bbox_size/2), int(bbox_size/2), 3), dtype=np.uint8)
                    rh_label.fill(255)
                    rh_image = np.zeros((int(bbox_size/2), int(bbox_size/2), 3), dtype=np.uint8)
                    rh_image.fill(255)
            else:
                scale_n, translate_n = hand_utils.resize_scale(real_img)
                real_img = hand_utils.fix_image(scale_n, translate_n, real_img)
                lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(real_img, fix_coords=True)
                hand_utils.display_single_hand_skleton(hsk_frame, lfpts)
                hand_utils.display_single_hand_skleton(hsk_frame, rfpts)
                lx, ly, lw = hand_utils.get_mid(lfpts, bbox_size)
                rx, ry, rw = hand_utils.get_mid(rfpts, bbox_size)
                if lw != 0:
                    lh_label = hsk_frame[ly:ly+lw, lx:lx+lw, :]
                    lh_image = real_img[ly:ly+lw, lx:lx+lw, :]
                else:
                    lh_label = np.zeros((bbox_size, bbox_size, 3), dtype=np.uint8)
                    lh_label.fill(255)
                    lh_image = np.zeros((bbox_size, bbox_size, 3), dtype=np.uint8)
                    lh_image.fill(255)
                
                if rw != 0:
                    rh_label = hsk_frame[ry:ry+rw, rx:rx+rw, :]                
                    rh_image = real_img[ry:ry+rw, rx:rx+rw, :]
                else:
                    rh_label = np.zeros((bbox_size, bbox_size, 3), dtype=np.uint8)
                    rh_label.fill(255)
                    rh_image = np.zeros((bbox_size, bbox_size, 3), dtype=np.uint8)
                    rh_image.fill(255)
            

            losses, generated = model(Variable(data['label']), Variable(data['next_label']), Variable(data['image']), \
                    Variable(data['next_image']), Variable(cond_zeros), lh_label, lh_image, rh_label, rh_image, bbox_size, infer=True)

            # if total_steps % 100 == 0:
            #     gen_img = util.tensor2im(generated[0].data[0])[:,:1024,:]
            #     gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
            #     targets = torch.cat((data['image'], data['next_image']), dim=3)
            #     real_img = util.tensor2im(targets[0])[:,:1024,:]
            #     real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
            #     lhpts_gen, rhpts_gen = hand_utils.get_keypoints(gen_img)
            #     lhpts_gen = hand_utils.rescale_points(1024, 512, lhpts_gen)
            #     rhpts_gen = hand_utils.rescale_points(1024, 512, rhpts_gen)
            #     lhpts_real, rhpts_real = hand_utils.get_keypoints(real_img)
            #     lhpts_real = hand_utils.rescale_points(1024, 512, lhpts_real)
            #     rhpts_real = hand_utils.rescale_points(1024, 512, rhpts_real)
            #     hand_utils.display_hand_skleton(gen_img, lhpts_gen, rhpts_gen)
            #     hand_utils.display_hand_skleton(real_img, lhpts_real, rhpts_real)
            #     cv2.imwrite("tmp/out_gen_"+str(i)+"_"+str(epoch)+".png", gen_img)
            #     cv2.imwrite("tmp/out_real_"+str(i)+"_"+str(epoch)+".png", real_img)

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + (loss_dict['D_lhand_real'] + loss_dict['D_lhand_fake']) * 0.5 + (loss_dict['D_rhand_real'] + loss_dict['D_rhand_fake']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] # + loss_dict['G_GANface']

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
            if total_steps % 100 == 0:
                syn_img_hand = util.tensor2im(generated[0].data[0])
                height_s, width_s, channels_s = syn_img_hand.shape
                syn_img_hand = cv2.cvtColor(syn_img_hand[:,:int(width/2),:], cv2.COLOR_RGB2BGR)
                real_hand_img = real_img.copy()
                inputs = torch.cat((data['label'], data['next_label']), dim=3)
                input_label = util.tensor2im(inputs[0])[:,:int(width/2),:]
                input_label = cv2.cvtColor(input_label, cv2.COLOR_RGB2BGR)
                if opt.netG == "global":
                    scale_n, translate_n = hand_utils.resize_scale(input_label, myshape=(256, 512, 3))
                    input_label = hand_utils.fix_image(scale_n, translate_n, input_label, myshape=(256, 512, 3))
                else:
                    scale_n, translate_n = hand_utils.resize_scale(input_label)
                    input_label = hand_utils.fix_image(scale_n, translate_n, input_label)
                if opt.hand_discrim:
                    handsk_fake_label = cv2.hconcat([generated[4], generated[6]])
                    handsk_fake_image = cv2.hconcat([generated[5], generated[7]])
                    handsk_fake = cv2.vconcat([handsk_fake_label, handsk_fake_image])
                    syn_img_hand[:handsk_fake.shape[0], :handsk_fake.shape[1], :] = handsk_fake
                    handsk_real_label = cv2.hconcat([lh_label, rh_label])
                    handsk_real_image = cv2.hconcat([lh_image, rh_image])
                    handsk_real = cv2.vconcat([handsk_real_label, handsk_real_image])
                    real_hand_img[:handsk_real.shape[0], :handsk_real.shape[1], :] = handsk_real
                #print(syn_img_hand.shape, real_hand_img.shape, input_label.shape)
                output_image = cv2.hconcat([syn_img_hand, real_hand_img, input_label])
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
