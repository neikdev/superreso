import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_utils import Discriminator, Generator

# Hyper-parameters
lr = 0.0005
batch_size = 60
image_size = 64
color_channels = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(root='dataset/', train=True, transform=my_transforms, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create discriminator and generator networks
netD = Discriminator(color_channels, features_d).to(device)
netG = Generator(channels_noise, color_channels, features_g).to(device)

# Setup optimizers
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

netD.train()
netG.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(60, channels_noise, 1, 1).to(device)

writer_real = SummaryWriter(f"runs/GAN_MNIST/test_real")
writer_fake = SummaryWriter(f"runs/GAN_MNIST/test_fake")
step = 0

print('Starting training...')

for epoch in range(1, num_epochs + 1):
    for batch_index, (data, targets) in enumerate(data_loader):
        data = data.to(device)
        batch_size = data.shape[0]
        # 1/ Train Discriminator: max log(D(x)) + log(1 - D(G(z)))

        # Train on real image
        netD.zero_grad()
        label = (torch.ones(batch_size) * 0.9).to(device)
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()

        # Train on fake image
        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size) * 0.1).to(device)
        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # 2/ Train Generator: max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if batch_index % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_index}/{len(data_loader)}] \
                LossD: {lossD:.4f} LossG: {lossG:.4f}')
            with torch.no_grad():
                fake = netG(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Mnist Real Images', img_grid_real, global_step=step)
                writer_fake.add_image('Mnist Fake Images', img_grid_fake, global_step=step)

            step += 1
