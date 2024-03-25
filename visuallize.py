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

# Kvasir
img = Image.open(r"D:\research\MAPR_src\Kvasir-SEG\images\cju0qkwl35piu0993l0dewei2.jpg")
mask = Image.open(r"D:\research\MAPR_src\Kvasir-seg\masks\cju0qkwl35piu0993")