import torch
from torch import nn, Tensor
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Sigmoid

class ConvAE(nn.Module):
    def __init__(self, bottleneck: int = 16) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.name = f"Convolutional_{bottleneck}"
        self.log()

        self.encoder = nn.Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(32, self.bottleneck, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
        
        self.decoder = nn.Sequential(
            ConvTranspose2d(self.bottleneck, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLU(),
            ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLU(),
            ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def log(self) -> None:
        n = self.bottleneck * 8 * 8
        print(f"Compressed to {n} real values.")
