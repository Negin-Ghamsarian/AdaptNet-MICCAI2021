# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:11:39 2020

@author: Negin
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from losses_Binary import DiceLoss, IoULoss
from PIL import Image
import numpy as np
import os



def eval_dice_IoU(net, loader, device, test_counter, save_dir, save=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    dice = []
    IoU = []
    
    dice_coeff = DiceLoss()
    jaccard_index = IoULoss()

    try:
       os.mkdir(save_dir)
    except OSError:
       pass

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, name = batch['image'], batch['mask'], batch['name']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                dice.append(dice_coeff(pred, true_masks).item())
                IoU.append(jaccard_index(pred, true_masks).item())

            if save: #test_counter//10 == test_counter/10:
                probs = pred.squeeze(0)

                tf = transforms.Compose(
                  [
                   transforms.ToPILImage(),
                   transforms.Resize(512),
                   transforms.ToTensor()
                  ]
                  )

                probs = tf(probs.cpu())
                full_mask = Image.fromarray(probs.squeeze().cpu().numpy()*255).convert('RGB')
                full_mask = full_mask.save(save_dir+ str(name[0]) + '_' + str(test_counter) + '.png')

            pbar.update()

    net.train()
    return sum(dice)/n_val, np.std(dice), sum(IoU)/n_val, np.std(IoU), min(dice), min(IoU), max(dice), max(IoU)