import torch.nn as nn
import torch
from torch import Tensor

class ConvNoiseAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ConvNoiseAE"

        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.relu(self.enc_conv1(x))
        enc2 = self.relu(self.enc_conv2(enc1))
        enc3 = self.relu(self.enc_conv3(enc2))

        dec1 = self.relu(self.dec_conv1(enc3))
          # Skip connection 1.
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec2 = self.relu(self.dec_conv2(dec1))
          # Skip connection 2.
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec3 = self.sigmoid(self.dec_conv3(dec2))

        return dec3
