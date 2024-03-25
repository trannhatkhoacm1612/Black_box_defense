import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from untils import transform, target_transform, noise_transform
from torchvision import transforms

from dataset import get_dataset
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from architech import get_architech
from untils import AverageMeter, dice_coeff
import argparse

from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default="kvasir", type=str)
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--classifier", default=None) # path to checkpoint
parser.add_argument("--encoder", default="encoder", type=str) # path to checkpoint
parser.add_argument("--pretrained_encoder", default=None, type=str) # path to checkpoint
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--outdir", default=None) # path to outdir
parser.add_argument("--epochs", default=100, type=int)
args = parser.parse_args()

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


# def loss_encoder(noise_image, noise_image_pred, target, image, lamda):

#     criterion = nn.BCEWithLogitsLoss()
#     larange_multiflier = nn.MSELoss()
    
#     return - criterion(noise_image_pred, target) + lamda * (0.002 * 128 - larange_multiflier(noise_image, image))

    
    


# set up
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


# ------------------------------ Data loader ---------------------------------------

train_dataset = get_dataset("kvasir", "train")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

test_dataset = get_dataset("kvasir", "test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
# ----------------------------- Model loader --------------------------
model = get_architech("unetsmall")

model.cuda()
model.load_state_dict(torch.load(args.classifier))
requires_grad_(model, False)

# -------------------------------- Encoder Loader ----------------------
encoder = get_architech("encoder")
if args.pretrained_encoder:
    encoder.load_state_dict(torch.load(args.pretrained_encoder))
encoder.cuda()   

criterion = nn.BCEWithLogitsLoss()

optimizer = Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)



def train(loader, model, criterion, encoder, optimizer, epochs, normalize=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))):
    model.eval()
    encoder.train()
    
    train_loss_meter = AverageMeter()
    # dice_meter = AverageMeter()
    train_loss_original_meter = AverageMeter()
    
    # lamda = torch.tensor(0.05, requires_grad=True)

    for epoch in tqdm(range(epochs)):
        check = None
        
        train_loss_meter.reset()
        # dice_meter.reset()
        # train_loss_original_meter.reset()

        for (imgs, masks) in loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            noise_imgs = encoder(imgs)

            noise_imgs = imgs + 0.1 * noise_imgs
            check = noise_imgs[0]

            noise_imgs = normalize(noise_imgs)
                        
            output_noise = model(noise_imgs)
            
            # loss = loss_encoder(noise_imgs, output_noise, masks, normalize(imgs), lamda)            
            loss = - criterion(output_noise, masks)
                        
            train_loss_meter.update(loss, imgs.shape[0])
            # train_loss_original_meter.update(criterion(output_noise, masks), imgs.shape[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        
        if (epoch + 1) % 10 == 0:
            print(f"train loss with custom loss fuction: {train_loss_meter.avg}")
            # print(f"train loss with original loss fuction: {train_loss_original_meter.avg}")
            # print(f"train with custom loss: {train_loss_meter.avg}")
            save_image(check,  os.path.join(args.outdir, f"{epoch + 1}.jpg"))
            torch.save(encoder.state_dict(), os.path.join(args.outdir, f"model_ep_{epoch + 1}.pth"))

def test(loader, model, criterion, encoder, normalize=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))):
    model.eval()
    encoder.eval()
    
    test_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    test_loss_meter.reset()
    dice_meter.reset()
    
    with torch.no_grad():
        for (imgs, masks) in tqdm(loader):
            imgs, masks = imgs.cuda(), masks.cuda()
            noise_imgs = encoder(imgs)

            noise_imgs = imgs + 0.1 * noise_imgs
            save_image(imgs[0],  os.path.join(args.outdir, "test_original.jpg"))
            save_image(noise_imgs[0],  os.path.join(args.outdir, "test.jpg"))

            noise_imgs = normalize(noise_imgs)
            output_noise = model(noise_imgs)
            output_masks = output_noise.sigmoid().round()
            
            loss = criterion(output_noise, masks)
            dice_score = dice_coeff(output_masks, masks)
               
            test_loss_meter.update(loss, imgs.shape[0])
            dice_meter.update(dice_score, imgs.shape[0])

            
    print(f"test loss: {test_loss_meter.avg}")
    print(f"dice score: {dice_meter.avg}")

# train(train_loader, model, criterion, encoder, optimizer, args.epochs)
test(test_loader, model, criterion, encoder) 