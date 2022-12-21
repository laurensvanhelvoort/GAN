import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Hyperparameters
lr = 3e-4  # Learning rate
z_dim = 64  # 128, 256, 32 also possible
image_dim = 28 * 28 * 1  # 784
batch_size = 64
num_epochs = 400

# Initialize GAN components
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    # Mean and Std for MNIST dataset:0.1307, 0.3081 but 0.5 works better
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Create dataset
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss between input and target probabilities

# TensorBoard
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")  # Output fake images
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")  # Output the real images
step = 0

for epoch in range(num_epochs):
    for batch_index, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(real)) + log(1 - D(G(z))
        noise = torch.randn(batch_size, z_dim).to(device)  # Random Gaussian
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))  # log(D(real))

        fake = gen(noise)
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        # later
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) --> max log(D(G(z)) better
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # TensorBoard
        if batch_index == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Generated Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )

                step += 1
