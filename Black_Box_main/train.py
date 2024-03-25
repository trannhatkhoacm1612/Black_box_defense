import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from untils import transform, target_transform, noise_transform
from torchvision import transforms
import itertools
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from dataset import get_dataset
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from architech import get_architech
from untils import AverageMeter, dice_coeff, requires_grad_
import argparse



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--noise_sd', default=0.05, type=float)
parser.add_argument("--q", default=192, type=int)
parser.add_argument("--noise_type", type=str)
parser.add_argument("--optimizer_method", default="FO", type=str)
parser.add_argument('--dataset', default="kvasir", type=str)
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--classifier", default=None) # path to checkpoint
parser.add_argument("--denoiser", default=None) # arch
parser.add_argument("--encoder", default=None) # arch
parser.add_argument("--pretrained_decoder", default=None) # path to checkpoint
parser.add_argument("--pretrained_encoder", default=None) # path to checkpoint
parser.add_argument("--decoder", default=None) # path to checkpoint
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--outdir", default=None) # path to outdir
parser.add_argument("--epochs", default=200, type=int)
args = parser.parse_args()

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


# set up
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.outdir:
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)


# ------------------------------ Data loader ---------------------------------------

train_dataset = get_dataset("kvasir", "train")
test_dataset = get_dataset("kvasir", "test")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# ----------------------------- Model loader --------------------------
model = get_architech("unetsmall")
model.cuda()
model.load_state_dict(torch.load(args.classifier))
requires_grad_(model, False)

criterion = nn.BCEWithLogitsLoss()

# ---------------------------- Denoiser
denoiser = get_architech("dncnn")

if args.denoiser:
    denoiser.load_state_dict(torch.load(args.denoiser))
denoiser.cuda()

optimizer = Adam(denoiser.parameters(), lr=1e-3, weight_decay=1e-4)


# Auto encoder
if args.encoder or args.decoder:
    encoder = get_architech("encoder")
    decoder = get_architech("decoder")

if args.pretrained_encoder:
    encoder.load_state_dict(torch.load(args.encoder))
if args.pretrained_decoder:
    decoder.load_state_dict(torch.load(args.decoder))

optimizer = Adam(denoiser.parameters(), lr=1e-3, weight_decay=1e-4)

# Optimizer
# if args.optimizer_method in ["FO", "ZO"]:
#     optimizer = Adam(denoiser.parameters(), lr=1e-3, weight_decay=1e-4)
# else:
#     optimizer = Adam(itertools.chain(denoiser.parameters(), encoder.parameters(), decoder.parameters()), lr=1e-3, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    



# def train_ae(train_loader, test_laoder, model, denoiser, encoder, decocder, criterion, optimizer, epochs=args.epochs, optimize_method=args.optimizer_method, mu_=0.005, normalize=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))):
# def train(train_loader, test_loader, model,  denoiser, encoder, decoder, criterion, optimizer, epochs=args.epochs, optimize_method=args.optimizer_method, mu_=0.005, noise_sd=args.noise_sd, q_=192, noise_type=args.noise_type, outdir=args.outdir):

def train_AE_DS(train_loader, test_loader, model,  denoiser, encoder, decoder, criterion, optimizer, epochs=args.epochs, optimize_method=args.optimize_method, mu_=0.005, noise_sd=args.noise_sd, q_=args.q, noise_type=args.noise_type, outdir=args.outdir):
    model.eval()
    denoiser.train()
    decoder.eval()
    encoder.eval()
    
    train_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    best_score = None
    
    for epoch in tqdm(range(epochs)):
        train_loss_meter.reset()
        dice_meter.reset()
        
        for (imgs, masks) in train_loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            
            if noise_type == "random":
                noise_imgs = imgs + torch.randn_like(imgs).cuda() * noise_sd
            
            elif noise_type == "laplace":
                gaussian_noise = torch.rand_like(imgs)
                laplace_noise = torch.distributions.Laplace(0, noise_sd).sample(gaussian_noise.shape).cuda()
                noise_imgs = imgs + laplace_noise
            
            denoised_imgs = denoiser(noise_imgs) # z = D(x)
            denoised_imgs = encoder(denoised_imgs) # z = D(x)
            
            if optimize_method == "FO":
                denoised_imgs = decoder(denoised_imgs)

                denoised_imgs = model(denoised_imgs)
                mask_preds = denoised_imgs.sigmoid().round()
                loss = criterion(denoised_imgs, masks)
                train_loss_meter.update(loss, imgs.shape[0])
#                 dice_score = dice_coeff(mask_preds, masks)
            
            elif optimize_method == "ZO":
                denoised_imgs.requires_grad_(True)
                denoised_imgs.retain_grad()
                batch_size = denoised_imgs.size()[0]
                channel = denoised_imgs.size()[1]
                h = denoised_imgs.size()[2]
                w = denoised_imgs.size()[3]
                d = channel * h * w
                
                with torch.no_grad():
                    m, sigma = 0, 100
                    mu = torch.tensor(mu_).cuda()
                    q = torch.tensor(q_).cuda()
                    
                    denoised_img_preds = model(decoder(denoised_imgs)) # a = f(z)
                    loss_0 = criterion(denoised_img_preds, masks) # L(a)
#                     print(loss_0)
                    train_loss_meter.update(loss_0, batch_size)
                    
                    denoised_imgs_flat_no_grad = torch.flatten(denoised_imgs, start_dim=1).cuda() #  
                    grad_est = torch.zeros(batch_size, d).cuda()
                    
                    for k in range(q_):
                        u = torch.normal(m, sigma, size=(batch_size, d))
                        u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d) # normalizer
                        u = torch.div(u, u_norm).cuda()
                        
                        denoised_q = denoised_imgs_flat_no_grad + mu * u
                        denoised_q = denoised_q.view(batch_size, channel, h, w)
                        denoised_q_pre = model(decoder(denoised_q)) # L(z + u.u_i)
                        
                        loss_tmp = criterion(denoised_q_pre, masks)
                        loss_diff = torch.tensor(loss_tmp - loss_0)
                        grad_est = grad_est + (d / q) * u * loss_diff / mu
                
                denoised_imgs_flat = torch.flatten(denoised_imgs, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()
                
                # loss = torch.sum(denoised_imgs_flat * grad_est_no_grad, dim=-1).mean()
                loss = torch.sum(denoised_imgs_flat * grad_est_no_grad)
#                 dice_score = dice_coeff(denoised_img_preds.sigmoid().round(), masks)                
                                     
                train_loss_meter.update(loss_0, imgs.shape[0])
#             dice_meter.update(dice_score, imgs.shape[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for (imgs, masks) in test_loader:
                imgs, masks = imgs.cuda(), masks.cuda()
                if noise_type == "random":
                    noise_imgs = imgs + torch.randn_like(imgs).cuda() * noise_sd
                elif noise_type == "laplace":
                    gaussian_noise = torch.rand_like(imgs)
                    laplace_noise = torch.distributions.Laplace(0, noise_sd).sample(gaussian_noise.shape).cuda()
                    noise_imgs = imgs + laplace_noise

                denoised_imgs = denoiser(noise_imgs) # z = D(x)
                denoised_imgs = encoder(denoised_imgs) # z = D(x)
                
                outputs = model(decoder(denoised_imgs))
                dice_score = dice_coeff(outputs.sigmoid().round(), masks)
                dice_meter.update(dice_score, imgs.shape[0])        
                                
        print(f"EP {epoch + 1}, train loss = {train_loss_meter.avg}\n dice score = {dice_meter.avg}")
        if not best_score or dice_meter.avg > best_score:
            best_score = dice_meter.avg
#         if (epoch + 1) % 20 == 0:
            torch.save(denoiser.state_dict(), os.path.join(outdir, f"best_denoiser_ZO.pth"))
            # torch.save(encoder.state_dict(), os.path.join(outdir, f"best_encoder_ZO.pth"))
            # torch.save(decoder.state_dict(), os.path.join(outdir, f"best_decode_ZO.pth"))
        print(f"Dice Score: {best_score}")

    
    
# optimizer = Adam(denoiser.parameters(), lr=1e-3, weight_decay=1e-4)



def train(loader, model, denoiser, criterion, optimizer, epochs=args.epochs, optimize_method=args.optimizer_method, mu_=0.005, std=(0.229, 0.224, 0.225)):
    model.eval()
    denoiser.train()
    
    train_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    for epoch in tqdm(range(epochs)):
        train_loss_meter.reset()
        dice_meter.reset()
        
        for (imgs, masks) in tqdm(loader):
            imgs, masks = imgs.cuda(), masks.cuda()
            
            if args.noise_type == "random":
                noise_imgs = imgs + torch.randn_like(imgs, device="cuda") * args.noise_sd
            
            elif args.noise_type == "laplace":
                gaussian_noise = torch.rand_like(imgs)
                laplace_noise = torch.distributions.Laplace(0, args.noise_sd).sample(gaussian_noise.shape).cuda()
                noise_imgs = imgs + laplace_noise
            
            denoised_imgs = denoiser(noise_imgs) # z = D(x)
            # denoised_imgs = normalize(denoised_imgs) # z = D(x)
            
            if optimize_method == "FO":
                denoised_imgs = model(denoised_imgs)
                mask_preds = denoised_imgs.sigmoid().round()
                loss = criterion(denoised_imgs, masks)
                dice_score = dice_coeff(mask_preds, masks)
            
            elif optimize_method == "ZO":
                denoised_imgs.requires_grad_(True)
                denoised_imgs.retain_grad()
                batch_size = denoised_imgs.size()[0]
                channel = denoised_imgs.size()[1]
                h = denoised_imgs.size()[2]
                w = denoised_imgs.size()[3]
                d = channel * h * w
                
                with torch.no_grad():
                    m, sigma = 0, 100
                    mu = torch.tensor(mu_).cuda()
                    q = torch.tensor(args.q).cuda()
                    
                    denoised_img_preds = model(denoised_imgs) # a = f(z)
                    loss_0 = criterion(denoised_img_preds, masks) # L(a)
                    train_loss_meter.update(loss_0, batch_size)
                    
                    denoised_imgs_flat_no_grad = torch.flatten(denoised_imgs, start_dim=1).cuda() #  
                    grad_est = torch.zeros(batch_size, d).cuda()
                    
                    for k in range(args.q):
                        u = torch.normal(m, sigma, size=(batch_size, d))
                        u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d) # normalizer
                        u = torch.div(u, u_norm).cuda()
                        
                        denoised_q = denoised_imgs_flat_no_grad + mu * u
                        denoised_q = denoised_q.view(batch_size, channel, h, w)
                        denoised_q_pre = model(denoised_q) # L(z + u.u_i)
                        
                        loss_tmp = criterion(denoised_q_pre, masks)
                        loss_diff = torch.tensor(loss_tmp - loss_0)
                        grad_est = grad_est + (d / q) * u * loss_diff / mu
                
                denoised_imgs_flat = torch.flatten(denoised_imgs, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()
                
                # loss = torch.sum(denoised_imgs_flat * grad_est_no_grad, dim=-1).mean()
                loss = torch.sum(denoised_imgs_flat * grad_est_no_grad)
                dice_score = dice_coeff(denoised_img_preds.sigmoid().round(), masks)                
                                     
            # train_loss_meter.update(loss, imgs.shape[0])
            dice_meter.update(dice_score, imgs.shape[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"EP {epoch + 1}, train loss = {train_loss_meter.avg}\n dice score = {dice_meter.avg}")
        if (epoch + 1) % 20 == 0:
            torch.save(denoiser.state_dict(), os.path.join(args.outdir, f"model_ep_{epoch + 1}.pth"))

def test(loader, model, denoiser, criterion, normalize=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))):
    
    model.eval()
    denoiser.eval()
    
    dice_meter = AverageMeter()
    dice_origninal_meter = AverageMeter()
    test_loss_meter = AverageMeter()
    with torch.no_grad():
        for (imgs, masks) in tqdm(loader):
            
            # dice_meter.reset()
            # dice_origninal_meter.reset()
            
            imgs, masks = imgs.cuda(), masks.cuda()
            
            # if args.noise_type == "random":
            #     noise_imgs = imgs + torch.randn_like(imgs, device="cuda") * args.noise_sd
                
            # elif args.noise_type == "laplace":
            #     gaussian_noise = torch.rand_like(imgs)
            #     laplace_noise = torch.distributions.Laplace(0, args.noise_sd).sample(gaussian_noise.shape).cuda()
            #     noise_imgs = imgs + laplace_noise
            
            noise_imgs = imgs + torch.randn_like(imgs, device="cuda") * args.noise_sd

            
            save_image(imgs[0], os.path.join(args.outdir, f"orignial.png"))
            save_image(noise_imgs[0], os.path.join(args.outdir, f"noise_{args.noise_type}.png"))
            
            
    #         dice_score_original = dice_coeff(model(normalize(noise_imgs)).sigmoid().round(), masks)
            
    #         noise_imgs = denoiser(noise_imgs) # denoiser

    #         noise_imgs = normalize(noise_imgs)

                    
    #         outputs = model(noise_imgs)
    #         output_masks = outputs.sigmoid().round()
            
    #         loss = criterion(outputs, masks)
    #         dice_score = dice_coeff(output_masks, masks)
            
            
    #         test_loss_meter.update(loss, imgs.shape[0])
    #         dice_meter.update(dice_score, imgs.shape[0])
    #         dice_origninal_meter.update(dice_score_original, imgs.shape[0])
                
    # print(f"Dice score denoiser: {dice_meter.avg} , test loss: {test_loss_meter.avg}")
    # print(f"Dice score with noise: {dice_origninal_meter.avg}")


# train(loader=train_loader, model=model, denoiser=denoiser, criterion=criterion, optimizer=optimizer)
test(test_loader, model, denoiser, criterion, normalize=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
