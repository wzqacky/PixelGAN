from model import *
from parse import *

import torch 
import matplotlib.pyplot as plt 
import torchvision.utils as vutils
import numpy as np
import sys 


device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


netG = Generator().to(device)
netG.apply(initialize_weights)
print(netG)

# load the pretrained model here 
parser = test_parser()
args = parser.parse_args(sys.argv[1:])
gen = torch.load(f'pretrained_model/{args.model}')
netG.load_state_dict(gen['model_state_dict'])

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
img_list = []
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
plt.show()