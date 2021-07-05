"""
@author: Negin Ghamsarian
"""

import os
from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import albumentations as A
import matplotlib.pyplot as plt


class BasicDataset_OneClass(Dataset):
    def __init__(self, imgs_dir, masks_dir, size=512, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.size = size
        self.mask_suffix = mask_suffix
        

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(os.listdir(self.imgs_dir))} examples')
        
        self.augmentation_pipeline = A.Compose([A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2), 
            A.RandomBrightnessContrast(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.Resize(self.size, self.size)
            ], p = 1)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_im(cls, img):

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if img.max() > 1:
            img = img / 255

        return img
    
    
    def preprocess_mask(cls, mask):
        
        if len(mask.shape) == 3:
           mask = mask[:,:,0]
        
        
        return mask
    
    

    def __getitem__(self, i):
        idx = self.ids[i]
        
            
        image = plt.imread(self.imgs_dir + '/' + idx + '.png')
        mask = plt.imread(self.masks_dir + '/' + idx + '.png')

        mask = self.preprocess_mask(mask)
        image = self.preprocess_im(image)
        image, mask = self.transform(image, mask)



        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor).permute(2,0,1),
            'mask': torch.from_numpy(mask).type(torch.long),
            'name': str(self.ids[i])
        }
        
        
    def transform(self, image, mask):

        transformed = self.augmentation_pipeline(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        return transformed_image, transformed_mask
         
         

