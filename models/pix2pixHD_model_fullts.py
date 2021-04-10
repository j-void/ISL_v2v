### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.hand_utils as hand_utils
import cv2
import torchvision.transforms as transforms
from PIL import Image

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.img_idx = 0
        ##### define networks        
        # Generator network
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1          
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 4*opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            
        if self.isTrain and self.opt.hand_discrim:
            use_sigmoid = opt.no_lsgan
            # self.netDhand = networks.define_D(6, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
            #                               1, False, gpu_ids=self.gpu_ids, netD='hand')
            self.netDhand = networks.define_D(6, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, False, gpu_ids=self.gpu_ids)

        
        self.data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
        
        print('---------- Networks initialized -------------')

        # load networks
        if (not self.isTrain or opt.continue_train or opt.load_pretrain):
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                if opt.hand_discrim:
                    self.load_network(self.netDhand, 'Dhand', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionHandGAN = torch.nn.BCELoss()
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            if opt.use_l1:
                self.criterionL1 = torch.nn.L1Loss()
        
            # Loss names
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'D_hand_fake', 'D_hand_real']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]                            
            else:
                params = list(self.netG.parameters())

            if opt.niter_fix_main == 0:
                params += list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D
            if opt.niter_fix_main > 0:
                print('------------- Only training hand discriminator network  ------------')
                params = list(self.netDhand.parameters())                         
            else:
                if opt.hand_discrim:
                    params = list(self.netD.parameters()) + list(self.netDhand.parameters())   
                else:
                    params = list(self.netD.parameters())   

            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, real_image=None, next_label=None, next_image=None, zeroshere=None, infer=False):

        input_label = label_map.data.float().cuda()
        input_label = Variable(input_label, volatile=infer)

        # next label for training
        if next_label is not None:
            next_label = next_label.data.float().cuda()
            next_label = Variable(next_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.float().cuda())

        # real images for training
        if next_image is not None:
            next_image = Variable(next_image.data.float().cuda())

        if zeroshere is not None:
            zeroshere = zeroshere.data.float().cuda()
            zeroshere = Variable(zeroshere, volatile=infer)


        return input_label, real_image, next_label, next_image, zeroshere

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate_4(self, s0, s1, i0, i1, use_pool=False):
        input_concat = torch.cat((s0, s1, i0.detach(), i1.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminateface(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netDface.forward(fake_query)
        else:
            return self.netDface.forward(input_concat)
        
    def discriminatehand(self, keypoints, use_pool=False):
        return self.netDhand.forward(keypoints)
    
    def discriminatehand_cgan(self, label, image, use_pool=False):
        input_concat = torch.cat((label, image.detach()), dim=1)
        return self.netDhand.forward(input_concat)

    def forward(self, label, next_label, image, next_image, zeroshere, hlabel_real, real_frame_cv, infer=False):
        # Encode Inputs
        input_label, real_image, next_label, next_image, zeroshere = self.encode_input(label, image, \
                     next_label=next_label, next_image=next_image, zeroshere=zeroshere)
                    
        hlabel_real_tensor = 0
        hlabel_fake_tensor = 0
        
        if self.opt.hand_discrim:
            hlabel_real_tensor = self.data_transforms(Image.fromarray(cv2.cvtColor(hlabel_real.copy(), cv2.COLOR_BGR2RGB)))
            hlabel_real_tensor = hlabel_real_tensor.view(1, hlabel_real.shape[2], hlabel_real.shape[0], hlabel_real.shape[1]).cuda()
        
        initial_I_0 = 0
        #print(input_label.size())
        # Fake Generation I_0
        input_concat = torch.cat((input_label, zeroshere), dim=1) 

        I_0 = self.netG.forward(input_concat)
        
        gen_img = util.tensor2im(I_0.data[0])
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
        
        hsk_frame = np.zeros(gen_img.shape, dtype=np.uint8)
        hsk_frame.fill(255)
        
        hand_frame_fake = np.zeros(gen_img.shape, dtype=np.uint8)
        hand_frame_fake.fill(255)
        
        hand_frame_real = np.zeros(gen_img.shape, dtype=np.uint8)
        hand_frame_real.fill(255)
            
        self.img_idx = self.img_idx + 1
        input_concat1 = torch.cat((next_label, I_0), dim=1)

        I_1 = self.netG.forward(input_concat1)

        loss_D_fake_face = loss_D_real_face = loss_G_GAN_face = 0
        fake_face_0 = fake_face_1 = real_face_0 = real_face_1 = 0
        fake_face = real_face = face_residual = 0
        
        loss_D_fake_hand = 0
        loss_D_real_hand = 0
        

                
        if self.opt.hand_discrim:
            
            if self.opt.netG == "global":
                scale_n, translate_n = hand_utils.resize_scale(gen_img, myshape=(256, 512, 3))
                gen_img = hand_utils.fix_image(scale_n, translate_n, gen_img, myshape=(256, 512, 3))
                lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(gen_img, fix_coords=True, sz=64)
                lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
                rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
                hand_frame_fake[lbx:lbx+lbw, lby:lby+lbw, :] = gen_img[lbx:lbx+lbw, lby:lby+lbw, :]
                hand_frame_fake[rbx:rbx+rbw, rby:rby+rbw, :] = gen_img[rbx:rbx+rbw, rby:rby+rbw, :]
                hand_frame_real[lbx:lbx+lbw, lby:lby+lbw, :] = real_frame_cv[lbx:lbx+lbw, lby:lby+lbw, :]
                hand_frame_real[rbx:rbx+rbw, rby:rby+rbw, :] = real_frame_cv[rbx:rbx+rbw, rby:rby+rbw, :]
                hand_utils.display_single_hand_skleton(hsk_frame, lfpts, sz=2)
                hand_utils.display_single_hand_skleton(hsk_frame, rfpts, sz=2)

            else:
                scale_n, translate_n = hand_utils.resize_scale(gen_img)
                gen_img = hand_utils.fix_image(scale_n, translate_n, gen_img)
                lfpts_rz, rfpts_rz, lfpts, rfpts = hand_utils.get_keypoints_holistic(gen_img, fix_coords=True)
                lbx, lby, lbw = hand_utils.assert_bbox(lfpts)
                rbx, rby, rbw = hand_utils.assert_bbox(rfpts)
                print(lbx, lby, lbw, rbx, rby, rbw)
                hand_frame_fake[lbx:lbx+lbw, lby:lby+lbw, :] = gen_img[lbx:lbx+lbw, lby:lby+lbw, :]
                hand_frame_fake[rbx:rbx+rbw, rby:rby+rbw, :] = gen_img[rbx:rbx+rbw, rby:rby+rbw, :]
                hand_frame_real[lbx:lbx+lbw, lby:lby+lbw, :] = real_frame_cv[lbx:lbx+lbw, lby:lby+lbw, :]
                hand_frame_real[rbx:rbx+rbw, rby:rby+rbw, :] = real_frame_cv[rbx:rbx+rbw, rby:rby+rbw, :]
                hand_utils.display_single_hand_skleton(hsk_frame, lfpts)
                hand_utils.display_single_hand_skleton(hsk_frame, rfpts)
        
            
            hand_frame_fake_tensor = self.data_transforms(Image.fromarray(cv2.cvtColor(hand_frame_fake.copy(), cv2.COLOR_BGR2RGB)))
            hand_frame_fake_tensor = hand_frame_fake_tensor.view(1, hand_frame_fake.shape[2], hand_frame_fake.shape[0], hand_frame_fake.shape[1]).cuda() 
            
            hand_frame_real_tensor = self.data_transforms(Image.fromarray(cv2.cvtColor(hand_frame_real.copy(), cv2.COLOR_BGR2RGB)))
            hand_frame_real_tensor = hand_frame_real_tensor.view(1, hand_frame_real.shape[2], hand_frame_real.shape[0], hand_frame_real.shape[1]).cuda() 
            
            hlabel_fake_tensor = self.data_transforms(Image.fromarray(cv2.cvtColor(hsk_frame.copy(), cv2.COLOR_BGR2RGB)))
            hlabel_fake_tensor = hlabel_fake_tensor.view(1, hsk_frame.shape[2], hsk_frame.shape[0], hsk_frame.shape[1]).cuda()
            
            #print("Fake: ", hlabel_fake_tensor.shape, I_0.shape)
            #print("Real: ", hlabel_real_tensor.shape, image.shape)
            
            pred_fake_hand = self.discriminatehand_cgan(hlabel_fake_tensor, hand_frame_fake_tensor)
            loss_D_fake_hand = self.criterionGAN(pred_fake_hand, False)
            pred_real_hand = self.discriminatehand_cgan(hlabel_real_tensor, hand_frame_real_tensor)
            loss_D_real_hand = self.criterionGAN(pred_real_hand, True)
                        
            # pred_fake_hand = self.discriminatehand_cgan(hlabel_fake_tensor, I_0)
            # loss_D_fake_hand = self.criterionGAN(pred_fake_hand, False)
            # pred_real_hand = self.discriminatehand_cgan(hlabel_real_tensor, image)
            # loss_D_real_hand = self.criterionGAN(pred_real_hand, True)
                

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate_4(input_label, next_label, I_0, I_1, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate_4(input_label, next_label, real_image, next_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, next_label, I_0, I_1), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG0 = self.criterionVGG(I_0, real_image) * self.opt.lambda_feat
            loss_G_VGG1 = self.criterionVGG(I_1, next_image) * self.opt.lambda_feat
            loss_G_VGG = loss_G_VGG0 + loss_G_VGG1 
            if self.opt.netG == 'global': #need 2x VGG for artifacts when training local
                loss_G_VGG *= 0.5
            # if self.opt.hand_discrim:
            #     loss_G_VGG += 0.5 * self.criterionVGG(lh_image_fake_tensor, lh_image_real_tensor) * self.opt.lambda_feat
            #     loss_G_VGG += 0.5 * self.criterionVGG(rh_image_fake_tensor, rh_image_real_tensor) * self.opt.lambda_feat

        if self.opt.use_l1:
            loss_G_VGG += (self.criterionL1(I_1, next_image)) * self.opt.lambda_A
        
        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_D_fake_hand, loss_D_real_hand], None if not infer else [torch.cat((I_0, I_1), dim=3), fake_face, face_residual, initial_I_0, hsk_frame, hand_frame_fake, hand_frame_real] ]

    def inference(self, label, prevouts):

        # Encode Inputs        
        input_label, _, _, _, prevouts = self.encode_input(Variable(label), zeroshere=Variable(prevouts), infer=True)

        """ new face """
        I_0 = 0
        # Fake Generation

        input_concat = torch.cat((input_label, prevouts), dim=1) 
        initial_I_0 = self.netG.forward(input_concat)

        return initial_I_0

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.opt.hand_discrim:
            self.save_network(self.netDhand, 'Dhand', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())     
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_fixed_params_netD(self):
        params = list(self.netD.parameters()) #+ list(self.netDface.parameters())         
        self.optimizer_D = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning multiscale discriminator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
