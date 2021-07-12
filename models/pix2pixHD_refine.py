import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms

class Pix2PixHDModelRefine(BaseModel):
    def name(self):
        return 'Pix2PixHDModelRefine'
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.img_idx = 0
        ##### define networks        
        # Generator network
        netG_input_nc = opt.output_nc
        if not opt.no_instance:
            netG_input_nc += 1          
        self.netGrefine = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netDrefine = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                                opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            
        
        
        self.data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
        
        print('---------- Networks initialized -------------')

        # load networks
        if (not self.isTrain or opt.continue_train or opt.load_pretrain):
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netGrefine, 'Grefine', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netDrefine, 'Drefine', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            if opt.use_l1:
                self.criterionL1 = torch.nn.L1Loss()
        
            # Loss names
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake',\
                'G_GAN_hand_left', 'D_hand_left_real', 'D_hand_left_fake', 'G_GAN_hand_right', 'D_hand_right_real', 'D_hand_right_fake']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netGrefine.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]                            
            else:
                params = list(self.netGrefine.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D
            params = list(self.netDrefine.parameters())  
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
    
    def encode_input(self, real_image=None, input_image=None,zeroshere=None, infer=False):
    
        if real_image is not None:
            real_image = Variable(real_image.data.float().cuda())
            
        if input_image is not None:
            input_image = Variable(input_image.data.float().cuda())

        if zeroshere is not None:
            zeroshere = zeroshere.data.float().cuda()
            zeroshere = Variable(zeroshere, volatile=infer)


        return real_image, input_image, zeroshere
    
    def discriminate(self, input_label, test_image=None, use_pool=False):
        if test_image != None:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        else:
            input_concat = input_label.detach();
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netDrefine.forward(fake_query)
        else:
            return self.netDrefine.forward(input_concat)
    
    def forward(self, image, input_image, zeroshere, hand_bbox, bbox_size, infer=False):
        # Encode Inputs
        real_image, input_image, _ = self.encode_input(real_image=image, input_image=input_image, zeroshere=None)
                    
        
        initial_I_0 = 0

        #input_concat = torch.cat((input_label, zeroshere), dim=1) 
        

        I_0 = self.netGrefine.forward(input_image)
        
            
        self.img_idx = self.img_idx + 1
        
        loss_D_fake = 0
        loss_D_real = 0
        loss_G_GAN = 0
        
        hand_left_out = hand_right_out = 0
        
        loss_D_fake_hand_right = loss_D_fake_hand_left = loss_D_real_hand_right = loss_D_real_hand_left = loss_G_GAN_hand_left = loss_G_GAN_hand_right = 0
        
        if self.opt.refine_hand:
            hand_left_real = torch.zeros(input_image.shape[0], input_image.shape[1], bbox_size, bbox_size).cuda()
            hand_left_fake = torch.zeros(input_image.shape[0], input_image.shape[1], bbox_size, bbox_size).cuda()
            hand_right_real = torch.zeros(input_image.shape[0], input_image.shape[1], bbox_size, bbox_size).cuda()
            hand_right_fake = torch.zeros(input_image.shape[0], input_image.shape[1], bbox_size, bbox_size).cuda()
            if hand_bbox[4] > 0:
                _hand_left_real = input_image[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
                hand_left_real[:,:,:_hand_left_real.shape[2],:_hand_left_real.shape[3]] = _hand_left_real
                _hand_left_fake = I_0[:, :, hand_bbox[1]:hand_bbox[1]+bbox_size, hand_bbox[0]:hand_bbox[0]+bbox_size]
                hand_left_fake[:,:,:_hand_left_fake.shape[2],:_hand_left_fake.shape[3]] = _hand_left_fake
            if hand_bbox[5] > 0:
                _hand_right_real = input_image[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
                hand_right_real[:,:,:_hand_right_real.shape[2],:_hand_right_real.shape[3]] = _hand_right_real
                _hand_right_fake = I_0[:, :, hand_bbox[3]:hand_bbox[3]+bbox_size, hand_bbox[2]:hand_bbox[2]+bbox_size]
                hand_right_fake[:,:,:_hand_right_fake.shape[2],:_hand_right_fake.shape[3]] = _hand_right_fake
                
            pred_fake_pool_left = self.discriminate(input_label=hand_left_fake, test_image=None, use_pool=True)
            loss_D_fake_hand_left = self.criterionGAN(pred_fake_pool_left, False)        
            
            pred_real_left = self.discriminate(input_label=hand_left_real, test_image=None)
            loss_D_real_hand_left = self.criterionGAN(pred_real_left, True)
            
            pred_fake_left = self.netDrefine.forward(hand_left_fake)        
            loss_G_GAN_hand_left = self.criterionGAN(pred_fake_left, True)
            
            pred_fake_pool_right = self.discriminate(input_label=hand_right_fake, test_image=None, use_pool=True)
            loss_D_fake_hand_right = self.criterionGAN(pred_fake_pool_right, False)        
            
            pred_real_right = self.discriminate(input_label=hand_right_real, test_image=None)
            loss_D_real_hand_right = self.criterionGAN(pred_real_right, True)
            
            pred_fake_right = self.netDrefine.forward(hand_right_fake)
            loss_G_GAN_hand_right = self.criterionGAN(pred_fake_right, True)
            
            hand_left_out = torch.cat((hand_left_fake, hand_left_real), dim=3)
            hand_right_out = torch.cat((hand_right_fake, hand_right_real), dim=3)
        else:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label=I_0, test_image=None, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

            # Real Detection and Loss        
            pred_real = self.discriminate(input_label=real_image, test_image=None)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netDrefine.forward(I_0)        
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
            loss_G_VGG = self.criterionVGG(I_0, real_image) * self.opt.lambda_feat
            if self.opt.refine_hand:
                loss_G_VGG += self.criterionVGG(hand_left_fake, hand_left_real) * self.opt.lambda_feat
                loss_G_VGG += self.criterionVGG(hand_right_fake, hand_right_real) * self.opt.lambda_feat
            if self.opt.netG == 'global': #need 2x VGG for artifacts when training local
                loss_G_VGG *= 0.5

        if self.opt.use_l1:
            loss_G_VGG += (self.criterionL1(I_0, real_image)) * self.opt.lambda_A
            if self.opt.refine_hand:
                loss_G_VGG += (self.criterionL1(hand_left_fake, hand_left_real)) * self.opt.lambda_A
                loss_G_VGG += (self.criterionL1(hand_right_fake, hand_right_real)) * self.opt.lambda_A
                
        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake,\
            loss_G_GAN_hand_left, loss_D_real_hand_left, loss_D_fake_hand_left, loss_G_GAN_hand_right, loss_D_real_hand_right, \
                loss_D_fake_hand_right] , None if not infer else [I_0, hand_left_out, hand_right_out] ]     
    
    
    def inference(self, input_image):
    
        # Encode Inputs        
        _, input_image , _ = self.encode_input(real_image=None, input_image=input_image, zeroshere=None, infer=True)
        
        I_0 = 0
        # Fake Generation

        I_0 = self.netGrefine.forward(input_image)

        return I_0
    
    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()
    
    def save(self, which_epoch):
        self.save_network(self.netGrefine, 'Grefine', which_epoch, self.gpu_ids)
        self.save_network(self.netDrefine, 'Drefine', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netGrefine.parameters())     
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr