import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img_path = r"D:\research\MAPR_src\Tumor-Brain-Dataset\Testing\glioma_tumor\glioma_tumor_7.jpg"
image = Image.open(img_path)
image = transform(image)

noise_sd = 0.05

noise_imgs = image + torch.randn_like(image) * noise_sd
save_image(noise_imgs, r"D:\research\MAPR_src\gausion_noise.png")

gaussian_noise = torch.rand_like(image)
laplace_noise = torch.distributions.Laplace(0, noise_sd).sample(gaussian_noise.shape)
noise_imgs = image + laplace_noise
save_image(noise_imgs, r"D:\research\MAPR_src\laplace_noise.png")