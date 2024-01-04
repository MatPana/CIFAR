import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import MSELoss
from models.ConvAE import ConvAE
from models.ConvNoiseAE import ConvNoiseAE
from models.LinearAE import LinearAE
from models.CompTrainer import CompTrainer
from models.NoiseTrainer import NoiseTrainer
from utils.DataPrep import DataPrep
from utils.utils import PSNR

device = "cpu"

dir = "data/train"

prep = DataPrep(dir)
data_loader = prep.get_dataloaders()

epochs = 10
learning_rate = 0.001

criterion = PSNR

autoencoder = ConvNoiseAE()
optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

for noise in [0.05]:
    trainer = NoiseTrainer(autoencoder, optimizer, data_loader, noise)
    trainer.train(epochs, criterion)
