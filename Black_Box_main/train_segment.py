import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from untils import transform, target_transform, noise_transform
import torch
import torch.nn as nn
from architech import get_architech
import argparse
import random
import csv
from dataset import KvasirDataset


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--noise_sd', default=0.5, type=float)
parser.add_argument('--dataset', default="custom", type=str)
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--classifier", default=None) # path to checkpoint
parser.add_argument("--denoiser", default=None) # path to checkpoint
parser.add_argument("--encoder", default=None) # path to checkpoint
parser.add_argument("--decoder", default=None) # path to checkpoint

args = parser.parse_args()

if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



# train_size = (256,256)
# batch_size = 32
# device = "cuda:0"
# lr = 1e-3
# epochs = 50

# transform = transforms.Compose([
#     transforms.Resize(train_size),
#     transforms.ToTensor(),  
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])

# target_transform = transforms.Compose([
#     transforms.Resize(train_size),
#     transforms.ToTensor()
# ])

train_dataset = KvasirDataset("/kaggle/input/split-set/train_paths.csv", transform=transform, target_transform=target_transform)
test_dataset = KvasirDataset("/kaggle/input/split-set/test_paths.csv", transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

model = UNetSmall().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# dice = Dice(num_classes=2, average="macro").to(device)

train_loss_meter = AverageMeter()
dice_meter = AverageMeter()

def train(model, train_loader, criterion, optimizer, epochs, batch_size):
    for epoch in tqdm(range(epochs)):
        train_loss_meter.reset()
        dice_meter.reset()
        for batch_id, (imgs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs,  targets = imgs.to(device), targets.to(device)
            tensor_pred = model(imgs)
            loss = criterion(tensor_pred, targets)
            loss.backward()
            optimizer.step()
            mask_pred = tensor_pred.sigmoid().round()
#             dice_score = dice(mask_pred.to(torch.int8), targets.to(torch.int8))
            dice_score = dice_coeff(mask_pred, targets)

            train_loss_meter.update(loss.item(), batch_size)
            dice_meter.update(dice_score, batch_size)        
        
        print(f"EP {epoch + 1}, train loss = {train_loss_meter.avg}, dice score = {dice_meter.avg}")
    
        if epoch + 1 >= 10:
            torch.save(model.state_dict(), f"/kaggle/working/model_ep_{epoch + 1}.pth")

            
train(model, train_loader, criterion, optimizer, epochs, batch_size)         