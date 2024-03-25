import torch
from torchvision import transforms

# Average estimate
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
# Redifine dices score for evaluation
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0 # smoothing rate

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)

train_size = (128, 128)

# transform
# transform = transforms.Compose([
#     transforms.Resize(train_size),
#     transforms.ToTensor(),  
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])

transform = transforms.Compose([
    transforms.Resize(train_size),
    transforms.ToTensor()
])
target_transform = transforms.Compose([
    transforms.Resize(train_size),
    transforms.ToTensor()
])


noise_transform = transforms.Compose([
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])