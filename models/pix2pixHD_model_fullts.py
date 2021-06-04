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
import torch.nn.functional as F

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
            

        if self.isTrain and opt.shand_dis:
            self.netDshand = networks.define_D(opt.output_nc*4, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids, netD='hand')
        
        if  self.opt.shand_gen:
            if opt.shandGtype == 'unet':
                self.shandGen = networks.define_G(opt.output_nc*2, opt.output_nc, 64, 'unet', 
                                          n_downsample_global=2, n_blocks_global=5, n_local_enhancers=0, 
                                          n_blocks_local=0, norm=opt.norm, gpu_ids=self.gpu_ids)
            elif opt.shandGtype == 'global':
                self.shandGen = networks.define_G(opt.output_nc*2, opt.output_nc, 64, 'global', 
                                      n_downsample_global=3, n_blocks_global=5, n_local_enhancers=0, 
                                      n_blocks_local=0, norm=opt.norm, gpu_ids=self.gpu_ids)
            else:
                raise('face generator not implemented!')
        
        self.data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
        
        print('---------- Networks initialized -------------')

        # load networks
        if (not self.isTrain or opt.continue_train or opt.load_pretrain):
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                if opt.shand_dis:
                    self.load_network(self.shandGen, 'Dshand', opt.which_epoch, pretrained_path)
            if opt.shand_gen:
                self.load_network(self.shandGen, 'Gshand', opt.which_epoch, pretrained_path)

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
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake',\
                'G_GAN_hand_left', 'D_hand_left_real', 'D_hand_left_fake', 'G_GAN_hand_right', 'D_hand_right_real',\
                    'D_hand_right_fake',]

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

            if opt.shand_gen:
                params = list(self.shandGen.parameters())
            else:
                if opt.niter_fix_main == 0:
                    params += list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D
            if opt.niter_fix_main > 0:
                print('------------- Only training hand discriminator network  ------------')
                params = []
                if opt.shand_dis:
                    params = params + list(self.netDshand.parameters())              
            else:
                params = list(self.netD.parameters())  
                if opt.shand_dis:
                    params = params + list(self.netDshand.parameters())

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
        
    def discriminate_4_hand(self, s0, s1, i0, i1, use_pool=False):
        input_concat = torch.cat((s0, s1, i0.detach(), i1.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netDshand.forward(fake_query)
        else:
            return self.netDshand.forward(input_concat)

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
    
    def discriminateshand(self, label, image, use_pool=False):
        input_concat = torch.cat((label, image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netDshand.forward(fake_query)
        else:
            return self.netDshand.forward(input_concat)

    def forward(self, label, next_label, image, next_image, zeroshere, hand_bbox, next_hand_bbox, bbox_size, infer=False):
        # Encode Inputs
        input_label, real_image, next_label, next_image, zeroshere = self.encode_input(label, image, \
                     next_label=next_label, next_image=next_image, zeroshere=zeroshere)
                    
        
        initial_I_0 = 0
        #print(input_label.size())
        # Fake Generation I_0
        input_concat = torch.cat((input_label, zeroshere), dim=1) 
        
        hand_size_left_0 = (bbox_size, bbox_size)
        hand_size_right_0 = (bbox_size, bbox_size)
        
        cond_zeros_hand = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
        
        hand_label_left_0 = hand_label_right_0 = 0
        #print(hand_bbox, bbox_size)
        if self.opt.shand_gen:
            hand_label_left_0 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
            if hand_bbox[4] > 0:
                _hand_label_left_0 = input_label[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
                hand_size_left_0 = (_hand_label_left_0.shape[2], _hand_label_left_0.shape[3])
                hand_label_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]] = _hand_label_left_0
            hand_label_right_0 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
            if hand_bbox[5] > 0:
                _hand_label_right_0 = input_label[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
                hand_size_right_0 = (_hand_label_right_0.shape[2], _hand_label_right_0.shape[3])
                hand_label_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]] = _hand_label_right_0

        I_hand_left_0 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
        I_hand_right_0 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
        if self.opt.shand_gen:
            initial_I_0 = self.netG.forward(input_concat)
            print(torch.cat((hand_label_left_0, cond_zeros_hand), dim=1).shape)
            I_hand_left_0 = self.shandGen.forward(torch.cat((hand_label_left_0, cond_zeros_hand), dim=1))
            I_hand_right_0 = self.shandGen.forward(torch.cat((hand_label_right_0, cond_zeros_hand), dim=1))
            I_0 = initial_I_0.clone()
            if hand_bbox[4] > 0:
                I_0[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size] = I_hand_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]]
            if hand_bbox[5] > 0:
                I_0[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size] = I_hand_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]]
        else:
            I_0 = self.netG.forward(input_concat)
        
            
        self.img_idx = self.img_idx + 1
        input_concat1 = torch.cat((next_label, I_0), dim=1)
        
        hand_size_left_1 = (bbox_size, bbox_size)
        hand_size_right_1 = (bbox_size, bbox_size)
        
        if self.opt.shand_gen:
            hand_label_left_1 = torch.zeros(next_label.shape[0], next_label.shape[1], bbox_size, bbox_size).cuda()
            if next_hand_bbox[4] > 0:
                _hand_label_left_1 = next_label[:, :, next_hand_bbox[1]:next_hand_bbox[1]+bbox_size, next_hand_bbox[0]:next_hand_bbox[0]+bbox_size]
                hand_size_left_1 = (_hand_label_left_1.shape[2], _hand_label_left_1.shape[3])
                hand_label_left_1[:,:,:hand_size_left_1[0],:hand_size_left_1[1]] = _hand_label_left_1
            hand_label_right_1 = torch.zeros(next_label.shape[0], next_label.shape[1], bbox_size, bbox_size).cuda()
            if next_hand_bbox[5] > 0:        
                _hand_label_right_1 = next_label[:, :, next_hand_bbox[3]:next_hand_bbox[3]+bbox_size, next_hand_bbox[2]:next_hand_bbox[2]+bbox_size]
                hand_size_right_1 = (_hand_label_right_1.shape[2], _hand_label_right_1.shape[3])
                hand_label_right_1[:,:,:hand_size_right_1[0],:hand_size_right_1[1]] = _hand_label_right_1

        I_hand_left_1 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
        I_hand_right_1 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
        if self.opt.shand_gen:
            initial_I_1 = self.netG.forward(input_concat1)
            _hand_left_0 = torch.zeros(I_0.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            if hand_bbox[4] > 0:
                _hand_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]] = I_0[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
            _hand_right_0 = torch.zeros(I_0.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            if hand_bbox[5] > 0:    
                _hand_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]] = I_0[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
            I_hand_left_1 = self.shandGen.forward(torch.cat((hand_label_left_1, _hand_left_0), dim=1))
            I_hand_right_1 = self.shandGen.forward(torch.cat((hand_label_right_1, _hand_right_0), dim=1))
            I_1 = initial_I_1.clone()
            if next_hand_bbox[4] > 0:
                I_1[:, :, next_hand_bbox[1]:next_hand_bbox[1]+bbox_size, next_hand_bbox[0]:next_hand_bbox[0]+bbox_size] = I_hand_left_1[:,:,:hand_size_left_1[0],:hand_size_left_1[1]]
            if next_hand_bbox[5] > 0:
                I_1[:, :, next_hand_bbox[3]:next_hand_bbox[3]+bbox_size, next_hand_bbox[2]:next_hand_bbox[2]+bbox_size] = I_hand_right_1[:,:,:hand_size_right_1[0],:hand_size_right_1[1]]
        else:
            I_1 = self.netG.forward(input_concat1)

        loss_D_fake_face = loss_D_real_face = loss_G_GAN_face = 0
        fake_face_0 = fake_face_1 = real_face_0 = real_face_1 = 0
        fake_face = real_face = face_residual = 0
                
           
        fake_hand_left_0 = fake_hand_left_1 = fake_hand_right_0 = fake_hand_right_1 = 0
        real_hand_left_0 = real_hand_left_1 = real_hand_right_0 = real_hand_right_1 = 0
        hand_left_out = hand_right_out = 0
        loss_D_fake_hand_right = loss_D_fake_hand_left = loss_D_real_hand_right = loss_D_real_hand_left = loss_G_GAN_hand_left = loss_G_GAN_hand_right = 0
        if self.opt.shand_dis:
            fake_hand_left_0 = torch.zeros(I_0.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            fake_hand_left_1 = torch.zeros(I_1.shape[0], I_1.shape[1], bbox_size, bbox_size).cuda()
            real_hand_left_0 = torch.zeros(real_image.shape[0], real_image.shape[1], bbox_size, bbox_size).cuda()
            real_hand_left_1 = torch.zeros(next_image.shape[0], next_image.shape[1], bbox_size, bbox_size).cuda()
            
            if next_hand_bbox[4] > 0 and hand_bbox[4] > 0:
                fake_hand_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]] = I_0[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
                fake_hand_left_1[:,:,:hand_size_left_1[0],:hand_size_left_1[1]] = I_1[:, :, next_hand_bbox[1]:next_hand_bbox[1]+bbox_size, next_hand_bbox[0]:next_hand_bbox[0]+bbox_size]
                real_hand_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]] = real_image[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
                real_hand_left_1[:,:,:hand_size_left_1[0],:hand_size_left_1[1]] = next_image[:, :, next_hand_bbox[1]:next_hand_bbox[1]+bbox_size, next_hand_bbox[0]:next_hand_bbox[0]+bbox_size]
                
            
            fake_hand_right_0 = torch.zeros(I_0.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            fake_hand_right_1 = torch.zeros(I_1.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            real_hand_right_0 = torch.zeros(real_image.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            real_hand_right_1 = torch.zeros(next_image.shape[0], I_0.shape[1], bbox_size, bbox_size).cuda()
            
            if next_hand_bbox[5] > 0 and hand_bbox[5] > 0:
                fake_hand_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]] = I_0[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
                fake_hand_right_1[:,:,:hand_size_right_1[0],:hand_size_right_1[1]] = I_1[:, :, next_hand_bbox[3]:next_hand_bbox[3]+bbox_size, next_hand_bbox[2]:next_hand_bbox[2]+bbox_size]
                real_hand_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]] = real_image[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
                real_hand_right_1[:,:,:hand_size_right_1[0],:hand_size_right_1[1]] = next_image[:, :, next_hand_bbox[3]:next_hand_bbox[3]+bbox_size, next_hand_bbox[2]:next_hand_bbox[2]+bbox_size]
                
            
            pred_fake_pool_left = self.discriminate_4_hand(hand_label_left_0, hand_label_left_1, fake_hand_left_0, fake_hand_left_1, use_pool=True)
            loss_D_fake_hand_left = self.criterionGAN(pred_fake_pool_left, False)
            
            pred_real_left = self.discriminate_4_hand(hand_label_left_0, hand_label_left_1, real_hand_left_0, real_hand_left_1)
            loss_D_real_hand_left = self.criterionGAN(pred_real_left, True)
            
            pred_fake_left = self.netD.forward(torch.cat((hand_label_left_0, hand_label_left_1, fake_hand_left_0, fake_hand_left_1), dim=1))
            loss_G_GAN_hand_left = self.criterionGAN(pred_fake_left, True)
            
            pred_fake_pool_right = self.discriminate_4_hand(hand_label_right_0, hand_label_right_1, fake_hand_right_0, fake_hand_right_1, use_pool=True)
            loss_D_fake_hand_right = self.criterionGAN(pred_fake_pool_right, False)
            
            pred_real_right = self.discriminate_4_hand(hand_label_right_0, hand_label_right_1, real_hand_right_0, real_hand_right_1)
            loss_D_real_hand_right = self.criterionGAN(pred_real_right, True)
            
            pred_fake_right = self.netD.forward(torch.cat((hand_label_right_0, hand_label_right_1, fake_hand_right_0, fake_hand_right_1), dim=1))
            loss_G_GAN_hand_right = self.criterionGAN(pred_fake_right, True)
            
            hand_left_out = torch.cat((fake_hand_left_0, real_hand_left_0, hand_label_left_0), dim=3)
            hand_right_out = torch.cat((fake_hand_right_0, real_hand_right_0, hand_label_right_0), dim=3)
        
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
            if self.opt.shand_dis:
                loss_G_VGG += 0.5 * self.criterionVGG(fake_hand_left_0, real_hand_left_0) * self.opt.lambda_feat
                loss_G_VGG += 0.5 * self.criterionVGG(fake_hand_left_1, real_hand_left_1) * self.opt.lambda_feat
                loss_G_VGG += 0.5 * self.criterionVGG(fake_hand_right_0, real_hand_right_0) * self.opt.lambda_feat
                loss_G_VGG += 0.5 * self.criterionVGG(fake_hand_right_1, real_hand_right_1) * self.opt.lambda_feat

        if self.opt.use_l1:
            loss_G_VGG += (self.criterionL1(I_1, next_image)) * self.opt.lambda_A
                
        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, \
             loss_G_GAN_hand_left, loss_D_real_hand_left, loss_D_fake_hand_left, loss_G_GAN_hand_right, loss_D_real_hand_right, \
                loss_D_fake_hand_right], None if not infer else [torch.cat((I_0, I_1), dim=3), fake_face, face_residual, initial_I_0, \
                    hand_left_out, hand_right_out] ]

    def inference(self, label, prevouts, hand_bbox, prev_hand_bbox, bbox_size):

        # Encode Inputs        
        input_label, _, _, _, prevouts = self.encode_input(Variable(label), zeroshere=Variable(prevouts), infer=True)
        
        I_0 = 0
        # Fake Generation

        input_concat = torch.cat((input_label, prevouts), dim=1) 
        initial_I_0 = self.netG.forward(input_concat)
        
        hand_size_left_0 = (bbox_size, bbox_size)
        hand_size_right_0 = (bbox_size, bbox_size)
        
        if self.opt.shand_gen:
            hand_label_left_0 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
            if hand_bbox[4] > 0:
                _hand_label_left_0 = input_label[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
                hand_size_left_0 = (_hand_label_left_0.shape[2], _hand_label_left_0.shape[3])
                hand_label_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]] = _hand_label_left_0
            
            hand_label_right_0 = torch.zeros(input_label.shape[0], input_label.shape[1], bbox_size, bbox_size).cuda()
            ##print(hand_bbox, bbox_size)
            if hand_bbox[5] > 0:
                _hand_label_right_0 = input_label[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
                hand_size_right_0 = (_hand_label_right_0.shape[2], _hand_label_right_0.shape[3])
                hand_label_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]] = _hand_label_right_0
                
            _hand_left_0 = torch.zeros(initial_I_0.shape[0], initial_I_0.shape[1], bbox_size, bbox_size).cuda()
            if prev_hand_bbox[4] > 0:
                _hand_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]] = initial_I_0[:, :, prev_hand_bbox[1]:prev_hand_bbox[1]+bbox_size, prev_hand_bbox[0]:prev_hand_bbox[0]+bbox_size]
            
            _hand_right_0 = torch.zeros(initial_I_0.shape[0], initial_I_0.shape[1], bbox_size, bbox_size).cuda()
            if prev_hand_bbox[5] > 0:    
                _hand_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]] = initial_I_0[:, :, prev_hand_bbox[3]:prev_hand_bbox[3]+bbox_size, prev_hand_bbox[2]:prev_hand_bbox[2]+bbox_size]
                
            I_hand_left_0 = self.shandGen.forward(torch.cat((hand_label_left_0, _hand_left_0), dim=1))
            I_hand_right_0 = self.shandGen.forward(torch.cat((hand_label_right_0, _hand_right_0), dim=1))
            I_0 = initial_I_0.clone()
            if hand_bbox[4] > 0:
                I_0[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size] = I_hand_left_0[:,:,:hand_size_left_0[0],:hand_size_left_0[1]]
            if hand_bbox[5] > 0:
                I_0[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size] = I_hand_right_0[:,:,:hand_size_right_0[0],:hand_size_right_0[1]]
            return I_0

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
        if self.opt.shand_dis:
            self.save_network(self.netDshand, 'Dshand', which_epoch, self.gpu_ids)
        if self.opt.shand_gen:
            self.save_network(self.shandGen, 'Gshand', which_epoch, self.gpu_ids)

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
