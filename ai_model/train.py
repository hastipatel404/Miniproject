import torch
from torch.utils.data import DataLoader
from dataset import RoomDataset
from models.generator import Generator
from models.discriminator import Discriminator
import torch.nn as nn
from torchvision.utils import save_image
import os

data_path = "../Dataset/ROOM_DATASET/train"

BATCH_SIZE = 4
EPOCHS = 20
LR = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RoomDataset(data_path)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    for input_img, target_img in loader:

        input_img = input_img.to(device)
        target_img = target_img.to(device)

        fake_img = G(input_img)

        # Train D
        real = D(input_img, target_img)
        fake = D(input_img, fake_img.detach())

        loss_D = (bce(real, torch.ones_like(real)) +
                  bce(fake, torch.zeros_like(fake))) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train G
        fake = D(input_img, fake_img)

        loss_G = bce(fake, torch.ones_like(fake)) + 100 * l1(fake_img, target_img)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1} | D: {loss_D.item()} | G: {loss_G.item()}")

    save_image(fake_img * 0.5 + 0.5, f"outputs/epoch_{epoch+1}.png")

torch.save(G.state_dict(), "checkpoints/generator.pth")
print("Training Done ✅")
