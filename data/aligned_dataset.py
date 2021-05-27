### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import glob
import joblib

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### label maps    
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')              
        self.label_paths = sorted(make_dataset(self.dir_label))
        

        ### real images
        if opt.isTrain or self.opt.shand_gen:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.image_paths = sorted(make_dataset(self.dir_image))
        
        # bbox_file = joblib.load(os.path.join(opt.dataroot, 'bbox_out.pkl'))
        
        # self.bbox_list = bbox_file["bbox_list"]
        # self.max_bbox = bbox_file["max_bbox"]
        self.dataset_size = len(self.label_paths) 
      
    def __getitem__(self, index):        
        ### label maps
        paths = self.label_paths
        label_path = paths[index]              
        label = Image.open(label_path).convert('RGB')        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        original_label_path = label_path

        image_tensor = next_label = next_image = face_tensor = handpts_real_tensor = handpts_fake_tensor = 0
        hand_bbox = hand_bbox_next = 0
        ### real images 
        if self.opt.isTrain or self.opt.shand_gen:
            image_path = self.image_paths[index]   
            image = Image.open(image_path).convert('RGB')    
            transform_image = get_transform(self.opt, params)     
            image_tensor = transform_image(image).float()

        #hand_bbox = self.bbox_list[index]

        is_next = index < len(self) - 1
        if self.opt.gestures:
            is_next = is_next and (index % 64 != 63)

        """ Load the next label, image pair """
        if is_next:

            paths = self.label_paths
            label_path = paths[index+1]              
            label = Image.open(label_path).convert('RGB')        
            params = get_params(self.opt, label.size)          
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            next_label = transform_label(label).float()
            
            #hand_bbox_next = self.bbox_list[index+1]
            
            if self.opt.isTrain or self.opt.shand_gen:
                image_path = self.image_paths[index+1]   
                image = Image.open(image_path).convert('RGB')
                transform_image = get_transform(self.opt, params)      
                next_image = transform_image(image).float()

        """ If using the face generator and/or face discriminator """
        # if self.opt.face_discrim or self.opt.face_generator:
        #     facetxt_path = self.facetext_paths[index]
        #     facetxt = open(facetxt_path, "r")
        #     face_tensor = torch.IntTensor(list([int(coord_str) for coord_str in facetxt.read().split()]))

        # input_dict = {'label': label_tensor.float(), 'image': image_tensor, 
        #               'path': original_label_path, 'face_coords': face_tensor,
        #               'next_label': next_label, 'next_image': next_image }
        
        """ If using for hand keypoints """
        
        
        # input_dict = {'label': label_tensor.float(), 'image': image_tensor, 
        #         'path': original_label_path, 'next_label': next_label, 'next_image': next_image ,
        #         'max_bbox': self.max_bbox, 'hand_bbox' : hand_bbox, 'next_hand_bbox': hand_bbox_next}
        
        input_dict = {'label': label_tensor.float(), 'image': image_tensor, 
                'path': original_label_path, 'next_label': next_label, 'next_image': next_image }
        
        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedDataset'