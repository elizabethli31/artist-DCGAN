import os
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets as dsets
from torchvision import transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from utils.helpers import make_noise, save, weights_init

# Remove corrupted images from scraping process
for filename in os.listdir('test/data'):
    if filename.endswith('.jpg'):
        try:
            img = PIL.Image.open('test/data/'+filename) 
            img.verify()
        except (IOError, SyntaxError) as e:
            os.remove('test2/data/'+filename)
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Preparing data
dataset = dsets.ImageFolder(root='test2',
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=0)

device = torch.device("cuda:0" if (torch.cuda.is_available() and 1) else "cpu")


# Generator and Discriminator
# Transformations based on DCGAN paper
class Generator(nn.Module):
    def __init__(self, alpha=0.2):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Get to training image size
            nn.ConvTranspose2d(128, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, alpha=0.2):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(alpha),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)

gen = Generator(128).to(device)
gen.apply(weights_init)

dis = Discriminator().to(device)
dis.apply(weights_init)

optimizer_dis = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_gen = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Binary cross entropy
criterion = nn.BCELoss()

def train(gen, dis, dat):
    # Real
    dis.zero_grad()
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), 1, dtype=torch.float, device=device)
    output = dis(real_cpu).view(-1)
    d_real = criterion(output, label)
    d_real.backward()

    # Fake
    noise = make_noise(b_size)
    fake = gen(noise)
    label.fill_(0)
    output = dis(fake.detach()).view(-1)
    d_fake = criterion(output, label)
    d_fake.backward()
    d_err = d_real + d_fake
    optimizer_dis.step()

    gen.zero_grad()
    label.fill_(1)
    output = dis(fake).view(-1)
    g_err = criterion(output, label)
    g_err.backward()
    optimizer_gen.step()
        

    return d_err, g_err

fixed_noise = make_noise(64)

G_losses = []
D_losses = []

for epoch in range(75):
    c=0
    # For each batch in the dataloader
    for x, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        c+=1

        d_err, g_err = train(gen, dis, data)

        # Output training stats
        if c % 4 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 75, c, len(dataloader),
                     d_err.item(), g_err.item()))

        G_losses.append(d_err.item())
        D_losses.append(g_err.item())

    img = gen(fixed_noise).detach().cpu()
    save(img, f"outputs/img{epoch+25}.jpg")