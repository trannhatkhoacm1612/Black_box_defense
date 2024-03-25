import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from untils import transform, target_transform, noise_transform
import torch
import torch.nn as nn
from architech import get_architech
import argparse
import random
import csv





# parser = argparse.ArgumentParser(description="Pytorch defense traning")

# parser.add_argument("--segment", default="unetsmall", type=str)
# parser.add_argument("--pretrained_segment", type=str, help="path to segment pretrained")
# parser.add_argument("--imgs_dir", type=str, help="path to images dir")
# parser.add_argument("--csv_file", type=str, help="path to images csv file")
# parser.add_argument("--out_dir", type=str, help="output dir of noise images")

# parser.add_argument("--gpus", default="0", type=str)



# args = parser.parse_args()
# device = f"cuda:{args.gpus}"

def get_dataset(dataset: str, split: str) -> Dataset:

    # --------------------------CUSTOM------------------------------------
    if dataset == "kvasir":      
        if split == "train":
            annotations_file = "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG/split_set/train_.csv"

        else:
            annotations_file = "/mlcv1/WorkingSpace/Personal/baotg/Khoa/Kvasir-SEG/split_set/test_.csv"

        return KvasirDataset(annotations_file, transform=transform, target_transform=transform)
    
    elif dataset == "kvasir_noise":
        if split == "train":
            annotations_file = ""
        else:
            annotations_file = ""
        return NoiseKvasirDataset(annotations_file, transform=transform, target_transform=transform)
        

# Custom dataset
class KvasirDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None):
        self.csv_file = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_path = self.csv_file.iloc[idx][0]

        mask_path = img_path.replace("images", "masks")
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        mask = mask.convert("L")
        mask = np.array(mask)
        mask[mask > 0] == 1
        mask = Image.fromarray(mask.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask


class NoiseKvasirDataset(Dataset):
    # Kvasir Dataset with noise and orignial predict
    def __init__(self, csv_path, noise_folder, mask_pred_folder, transform=None, target_transform=None):
        self.csv_file = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.noise_folder = noise_folder
        self.mask_pred_folder = mask_pred_folder
        
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_path = self.csv_file.iloc[idx][0]
        mask_path = img_path.replace("images", "masks")
        file_name = os.path.basename(img_path)
        noise_path = os.path.join(self.noise_folder, file_name)
        mask_pred_path = os.path.join(self.mask_pred_folder, file_name)
        image = Image.open(img_path)
        noise_image = Image.open(noise_path)
        
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_pred = np.array(Image.open(mask_pred_path).convert("L"))
        
#         mask[mask > 0] = 1
#         mask_pred[mask_pred > 0] = 1
        mask = Image.fromarray(mask.astype(np.uint8))
        mask_pred = Image.fromarray(mask_pred.astype(np.uint8))
        
        
        if self.transform:
            image = self.transform(image)
            noise_image = self.transform(noise_image)
        if self.target_transform:
            mask = self.target_transform(mask)
            mask_pred = self.target_transform(mask_pred)

        return image, noise_image, mask, mask_pred 




def split_data(IMG_DIR, ratio=0.8):
    
    '''
    Args:
        IMG_DIR: folder container image (not mask)
        ratio: ratio for spliting training and testing set
    Ouput:
        csv map image path files for training and testing set
    
    '''
    
    img_name_lists = os.listdir(IMG_DIR)
    
    train_nums = int(ratio * len(img_name_lists))
    train_indxs = random.sample(range(len(img_name_lists)), train_nums)
    test_indxs = set(range(len(img_name_lists))).difference(train_indxs)
    train_name_lists = [img_name_lists[i] for i in train_indxs]
    test_name_lists = [img_name_lists[i] for i in test_indxs]
    
    train_path_lists = [os.path.join(IMG_DIR, file_name) for file_name in train_name_lists]
    test_path_lists = [os.path.join(IMG_DIR, file_name) for file_name in test_name_lists]
    
    with open(args.csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for path in train_path_lists:
            writer.writerow([path])
    
    with open(args.csv_file.replace("train", "test"), 'w', newline='') as file:
        writer = csv.writer(file)
        for path in test_path_lists:
            writer.writerow([path])

def fgsm_attack(image,data_grad, epsilon=0.02):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def denorm(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def create_noise_images(csv_path, model, criterion, transform, target_transform, mode="train"):
    
    '''
    Args:
        csv_path: path_line of image
        model: loaded pretrain model
        criterion: for loss
        transform: for train img
        target_transform: for mask img
        mode: for train for test
        
    Ouput:
        Train/Test folder: contains noise image and original predicted
    '''
    
    # load file map
    csv_file = pd.read_csv(csv_path)
    output_folder = os.path.join(args.out_dir, mode)
    output_folder_noise = os.path.join(output_folder, "noise") # change
    output_folder_pred = os.path.join(output_folder, "mask_pred") # change
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)   
    if not os.path.exists(output_folder_noise):
        os.mkdir(output_folder_noise)
    if not os.path.exists(output_folder_pred):
        os.mkdir(output_folder_pred)
    
    # main process
    for indx in tqdm(range(len(csv_file.index))):
        
        # load path file
        img_path = csv_file.iloc[indx][0]
        mask_path = img_path.replace("images", "masks")
        file_name = os.path.basename(img_path)
        
        # read image
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = np.array(mask)
#         mask[mask > 0] == 1
        mask = Image.fromarray(mask.astype(np.uint8))
        
        # processing image
        image = transform(image)
        mask = target_transform(mask)
        
        # fgsm attack
        model.train() # train mode to take gradient
        image, mask = image.to(device).unsqueeze(0), mask.to(device).unsqueeze(0)
        image.requires_grad = True # add into computional graph
        mask_pred = model(image) # orignial predict
        loss = criterion(mask_pred, mask) # loss for calculate gradient
        img_denorm = denorm(image) # denormalize
        model.zero_grad() # queeze
        loss.backward() # backprop for cal gradient
        img_grad = image.grad.data # take gradinet
        img_noise = fgsm_attack(img_denorm, img_grad) # attack base one img'gradient
        
        save_image(mask_pred.sigmoid().round(), os.path.join(output_folder_pred, file_name)) # save predicted mask
        save_image(img_noise.squeeze(), os.path.join(output_folder_noise, file_name)) # sace noise image


# if __name__ == "__main__":

#     # split data
#     split_data(args.imgs_dir, 0.7)

#     # --------------------- Load segmentation pretrained -------------------
#     model = get_architech(args.segment)
#     model.load_state_dict(torch.load(args.pretrained_segment))
#     model = model.to(device)

#     # criterion
#     criterion = nn.BCEWithLogitsLoss()

#     create_noise_images(args.csv_file,  model, criterion, transform, target_transform, "train")
#     create_noise_images(args.csv_file.replace("train", "test"),  model, criterion, transform, target_transform, "test")