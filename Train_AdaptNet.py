"""
@author: Negin Ghamsarian
"""

import argparse
import logging
import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from net import AdaptNet as Net

from utils_Binary import BasicDataset_binary as BasicDataset
from utils_Binary import save_metrics
from utils_Binary import eval_dice_IoU_binary as eval_dice_IoU
from losses_Binary import DiceBCELoss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




dir_train_img = 'Dataset_lens/imgs/train'  
dir_train_mask = 'Dataset_lens/lens/train'

dir_test_img = 'Dataset_lens/imgs/test'
dir_test_mask = 'Dataset_lens/lens/test'

save_test = 'visualization_LensID/Lens/AdaptNet_lr002/'

dir_checkpoint = 'checkpoints/LensID/AdaptNet_lr003/'
csv_name = 'CSVs_AdaptNet/AdaptNet_Lens_lr002.csv'




def train_net(net,
              device,
              epochs=30,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_size = 512):

    TESTS = []
    train_dataset = BasicDataset(dir_train_img, dir_train_mask, img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    n_train = len(train_dataset)



    test_dataset = BasicDataset(dir_test_img, dir_test_mask, img_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    n_test = len(test_dataset)


    writer = SummaryWriter(comment=f'_iris_AdaptNet_lr_{lr}_BS_{batch_size}_SIZE_{img_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Test size:       {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {img_size}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.8)
    
    criterion = DiceBCELoss()

    test_counter = 1
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                imgs = batch['image']
                true_masks = batch['mask']
                
                               
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                masks_pred = net(imgs)
                loss_main = criterion(masks_pred, true_masks)
                loss = loss_main
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                (loss_main).backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val1, val2, val3, val4, val5, val6, val7, val8 = eval_dice_IoU(net, test_loader, device, test_counter, save_test, save=False)
                    
                    TESTS.append([val1, val2, val3, val4, val5, val6, val7, val8, epoch_loss])

                    test_counter = test_counter+1
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                         print("NOT IMPLEMENTED")
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val1))
                        logging.info('Validation IoU: {}'.format(val3))
                        writer.add_scalar('test/Dice', val1, global_step)
                        writer.add_scalar('test/IoU', val3, global_step)

                    writer.add_images('images', imgs, global_step)


        scheduler.step()
           
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            
    val1, val2, val3, val4, val5, val6, val17, val18 = eval_dice_IoU(net, test_loader, device, test_counter, save_test, save=True)
    save_metrics(TESTS, csv_name)
     

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--size', dest='size', type=float, default=512,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net = Net(n_classes=1, n_channels=3)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_size=args.size,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
