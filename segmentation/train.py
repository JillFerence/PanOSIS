"""
Code base from: https://github.com/deepmancer/unet-semantic-segmentation
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import Segmentation_Dataset
from unet_stdconv import UNet
from utils import create_mask, compute_iou, numerical_sort, display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'data/cityscapes/'
image_path = os.path.join(path, 'images/')
mask_path = os.path.join(path, 'masks/')
image_list = numerical_sort(os.listdir(image_path))
mask_list = numerical_sort(os.listdir(mask_path))

results_dir = "train-results"
os.makedirs(results_dir, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 16
LR = 0.001
B1, B2 = 0.9, 0.999

dataset = Segmentation_Dataset(image_path, mask_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(B1, B2))
criterion = nn.CrossEntropyLoss()

train_losses, train_mious = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_iou = []

    for batch in train_loader:
        imgs = batch["IMAGE"].to(device)
        masks = batch["MASK"].to(device).squeeze(1).long()

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        batch_ious = [compute_iou(create_mask(preds)[b], masks[b], preds.shape[1]) for b in range(imgs.size(0))]
        total_iou.append(np.nanmean(batch_ious))

    avg_loss = total_loss / len(train_loader.dataset)
    avg_iou = np.mean(total_iou)
    train_losses.append(avg_loss)
    train_mious.append(avg_iou)

    print(f"Epoch {epoch+1} Loss: {avg_loss}, mIoU: {avg_iou}")

    if (epoch + 1) % 10 == 0:
            IMG = batch["IMAGE"][0, :, : ,:].to(device).unsqueeze(0)
            MASK = batch["MASK"][0, :, :, :].to(device).unsqueeze(0)
            pred_mask = model.to(device)(IMG)
            display([IMG[0].cpu(), MASK[0].cpu(), create_mask(pred_mask).cpu()], epoch + 1, results_dir)


torch.save(model.state_dict(), os.path.join(results_dir, "unet_trained.pth"))
np.save(os.path.join(results_dir, "train_losses.npy"), train_losses)
np.save(os.path.join(results_dir, "train_mious.npy"), train_mious)

plt.clf()
plt.plot(train_losses, label='Train Loss')
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig(os.path.join(results_dir, "train_loss.png"))

plt.clf()
plt.plot(train_mious, label='Train mIoU')
plt.xlabel("Epoch"), plt.ylabel("mIoU")
plt.title("Training mIoU")
plt.legend()
plt.savefig(os.path.join(results_dir, "train_miou.png"))