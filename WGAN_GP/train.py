from model import *
import random 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import torchvision.utils as vutils
import numpy as np

seed = 42
random.seed(seed)
torch.manual_seed(seed)

nz = 100 # dimension of latent space 
lr = 1e-4
LAMBDA_GP = 10
disc_iteration = 5

# data preprocessing 
IMG_SIZE = 64
BATCH_SIZE = 64

transformations = transforms.Compose([
                                transforms.Resize(IMG_SIZE),
                                transforms.CenterCrop(IMG_SIZE),
                                transforms.RandomAdjustSharpness(sharpness_factor=2),
                                transforms.RandomAutocontrast(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

data = datasets.ImageFolder(root="../Dataset", transform=transformations)

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# Visualizing the dataset 
batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Dataset Images")
plt.imshow(
    np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

# weight initialization (DCGAN)
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Check the generator model
netG = Generator().to(device)
netG.apply(initialize_weights)
print(netG)

# Check the discriminator model
netD = Discriminator().to(device)
netD.apply(initialize_weights)
print(netD)

# criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0, 0.9))

# train the model 

G_losses = []
D_losses = []
iters = 0
epochs = 500 

len_ds = len(dataloader)
status_step = len_ds // 2

# For each epoch
for epoch in range(epochs):
    print("Epoch:", epoch+1)
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # train the discriminator 
        netD.zero_grad()

        real = data[0].to(device)
        b_size = real.size(0)
        
        for _ in range(disc_iteration):
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            disc_real = netD(real).reshape(-1)
            disc_fake = netD(fake).reshape(-1)
            
            # apply gradient penalty 
            eps = torch.rand((b_size, 1, 1, 1)).to(device)
            interpolated = eps * real + (1-eps) * fake 
            scores = netD(interpolated)
            grad = torch.autograd.grad(
                inputs=interpolated,
                outputs=scores,
                grad_outputs=torch.ones(scores.size()).to(device),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad = grad.view(grad.shape[0], -1)
            grad_penalty = ((grad.norm(2, dim=1)-1)**2).mean()

            errD = -(torch.mean(disc_real) - torch.mean(disc_fake)) + grad_penalty * LAMBDA_GP
            errD.backward(retain_graph=True)
            optimizerD.step()
                
        # train the generator 
        netG.zero_grad()
        output = netD(fake).reshape(-1)
        errG = -torch.mean(output)
        errG.backward()
        optimizerG.step()
        
        if i % status_step == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                % (epoch, epochs, i, len_ds,
                   errD.item(), errG.item()))

        # Save Losses for plotting graph
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        iters += 1

# saving the model if you want 
torch.save({
            'epoch': epochs,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            }, "generator.pt")
torch.save({
            'epoch': epochs,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            }, "discriminator.pt")
