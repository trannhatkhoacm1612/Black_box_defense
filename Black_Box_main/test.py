from tqdm import tqdm
from untils import transform, target_transform
import torch
import torch.nn as nn
from architech import get_architech
import argparse
from untils import AverageMeter, dice_coeff
from dataset import NoiseKvasirDataset 
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description="Pytorch defense traning")

parser.add_argument("--segment", default="unetsmall", type=str)
parser.add_argument("--pretrained_segment", type=str, help="path to segment pretrained")
parser.add_argument("--denoiser", default="dncnn", type=str)
parser.add_argument("--pretrained_denoiser", type=str, help="path to pretrained of denoiser")
parser.add_argument("--denoiser", default=None) # arch
parser.add_argument("--encoder", default=None) # arch
parser.add_argument("--pretrained_decoder", default=None) # path to checkpoint
parser.add_argument("--pretrained_encoder", default=None) # path to checkpoint
parser.add_argument("--decoder", default=None) # path to checkpoint
parser.add_argument("--imgs_dir", type=str, help="path to images dir")
parser.add_argument("--noise_dir", type=str, help="path to noise images dir")
parser.add_argument("--mask_pred_dir", type=str, help="path to prediction")
parser.add_argument("--csv_file", type=str, help="path to images csv file")

parser.add_argument("--gpus", default="0", type=str)

args = parser.parse_args()
device = f"cuda:{args.gpus}"


def test_fgsm(test_loader, model, criterion):
    # model = model.to(device)
    
    test_loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    for (imgs, noise_images, masks, mask_preds) in tqdm(test_loader):
        imgs, noise_images, masks, mask_preds = imgs.to(device), noise_images.to(device), masks.to(device), mask_preds.to(device)
        noise_preds = model(noise_images)
        mask_noise_preds = noise_preds.sigmoid().round()
        
        loss = criterion(noise_preds, masks)
        dice_score = dice_coeff(mask_noise_preds, masks)
        
        test_loss_meter.update(loss, imgs.shape[0])
        dice_meter.update(dice_score, imgs.shape[0])
        
    print(f"Loss: {test_loss_meter.avg} , Dice score with noise-orignial: {dice_meter.avg}")


def main():
    # load data
    test_dataset = NoiseKvasirDataset(args.csv_file, args.noise_dir, args.mask_pred_dir, transform, target_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # load model
    model = get_architech(args.segment)
    model.load_state_dict(torch.load(args.pretrained_segment))
    model = model.to(device)
    # criterion
    criterion = nn.BCEWithLogitsLoss()

    # denoiser
    denoiser = get_architech(args.denoiser)
    denoiser.load_state_dict(torch.load(args.pretrained_denoiser))
    denoiser.to(device).eval()
    
    if args.encoder or args.decoder:
        encoder = get_architech("encoder")
        decoder = get_architech("decoder")

    if args.pretrained_encoder:
        encoder.load_state_dict(torch.load(args.encoder))
    if args.pretrained_decoder:
        decoder.load_state_dict(torch.load(args.decoder))    
        test_fgsm(test_loader, model, criterion)
    
main()