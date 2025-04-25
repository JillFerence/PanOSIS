"""
Code base from: https://github.com/deepmancer/unet-semantic-segmentation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from dataloader import Segmentation_Dataset
from unet_stdconv import UNet
import re

from PIL import Image
import torchvision.transforms.functional as F

def display(display_list, epoch):

    # define prefix path
    prefix = "results/epoch_" + str(epoch) + "_"
    # cdefine paths for each gt img, gt mask and pred mask
    gt_img_path = prefix + "gt_img.png"
    gt_mask_path = prefix + "gt_mask.png"
    pred_mask_path = prefix + "pred_mask.png"
    # shove them in one array
    paths = [gt_img_path, gt_mask_path, pred_mask_path]

    # for each image
    for i in range(3):
        # turn off the axis
        plt.axis('off')
        plt.imshow(display_list[i].permute(1,2,0))
        # save to path
        plt.savefig(paths[i], bbox_inches='tight', pad_inches=0)

def create_mask(pred_mask):
    pred_mask = torch.argmax(pred_mask, dim=1).detach()
    return pred_mask

def numerical_sort(file_list):
    return sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = 'data/'
    image_path = os.path.join(path, 'images/')
    mask_path = os.path.join(path, 'masks/')

    image_list = numerical_sort(os.listdir(image_path))
    mask_list = numerical_sort(os.listdir(mask_path))
    image_list = [os.path.join(image_path, i) for i in image_list]
    mask_list = [os.path.join(mask_path, i) for i in mask_list]

    # make results dir
    try:
        os.mkdir("results")
    except FileExistsError:
        print("results folder already exists. continuing...")

    EPOCHS = 40
    BATCH_SIZE = 16
    LR = 0.001
    B1 = 0.9
    B2 = 0.999
  
    unet = UNet().to(device)

    dataloader = DataLoader(Segmentation_Dataset(image_path, mask_path), batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(unet.parameters(), lr=LR, betas=(B1, B2))
    losses = []
    for epoch in range(EPOCHS):
        epoch_losses = []
        for i, batch in enumerate(dataloader):
            images = batch['IMAGE'].to(device)
            masks = batch['MASK'].to(device)
            N, C, H, W = masks.shape
            masks = masks.reshape((N, H, W)).long()

            optimizer.zero_grad()
            
            outputs = unet(images)

            # Compute loss
            loss = criterion(outputs, masks)
            epoch_losses.append(loss.item() * images.size(0))

            # Backward pass
            loss.backward()

            # EQUI gradient clipping for gradient explosion
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            print(f'EPOCH#{epoch},\t Batch#{i},\t Loss:{loss.item()}')
        losses.append(np.mean(epoch_losses) / len(dataloader.dataset))

        if (epoch + 1) % 10 == 0:
            IMG = batch["IMAGE"][0, :, : ,:].to(device).unsqueeze(0)
            MASK = batch["MASK"][0, :, :, :].to(device).unsqueeze(0)
            pred_mask = unet.to(device)(IMG)
            display([IMG[0].cpu(), MASK[0].cpu(), create_mask(pred_mask).cpu()], epoch + 1)

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show() 
