import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import Segmentation_Dataset
from unet_equiconv import UNet
from utils import create_mask, compute_iou, numerical_sort, display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'data/one-shot-streets/'
image_path = os.path.join(path, 'images/')
mask_path = os.path.join(path, 'masks/')
image_list = numerical_sort(os.listdir(image_path))
mask_list = numerical_sort(os.listdir(mask_path))
results_dir = "test-results"
os.makedirs(results_dir, exist_ok=True)

dataset = Segmentation_Dataset(image_path, mask_path)
train_size = int(0.8 * len(dataset))
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = UNet().to(device)
model.load_state_dict(torch.load(os.path.join("train-results", "unet_trained.pth")))
model.eval()

test_losses, test_mious = [], []
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        imgs = batch["IMAGE"].to(device)
        masks = batch["MASK"].to(device).squeeze(1).long()
        preds = model(imgs)

        loss = criterion(preds, masks)
        test_losses.append(loss.item() * imgs.size(0))

        pred_mask = create_mask(preds)
        batch_ious = [compute_iou(pred_mask[b], masks[b], preds.shape[1]) for b in range(imgs.size(0))]
        test_mious.append(np.nanmean(batch_ious))

        if i == 0:
            display([imgs[0].cpu(), masks[0].cpu().squeeze(), pred_mask[0].cpu()], i, results_dir)

avg_test_loss = np.mean(test_losses) / len(test_loader.dataset)
avg_test_iou = np.mean(test_mious)

print(f"Avg Loss: {avg_test_loss}, Avg mIoU: {avg_test_iou}")