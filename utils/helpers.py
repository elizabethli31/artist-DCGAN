import torch
import torch.nn as nn
from torchvision.utils import save_image


# set the computation device
device = torch.device('cuda:0' if (torch.cuda.is_available() and 1 ) else 'cpu')

# Noise for generator
def make_noise(size):
    return torch.randn(size, 128, 1, 1).to(device)

def save(image, path):
    save_image(image, path, normalize=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

