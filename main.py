import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import MSELoss
from models.ConvAE import ConvAE
from models.LinearAE import LinearAE
from models.CompTrainer import CompTrainer
from utils.DataPrep import DataPrep
from utils.utils import PSNR

device = "cpu"

dir = "data/train"

prep = DataPrep(dir)
data_loader = prep.get_dataloaders()

# autoencoder = LinearAE([256 * 4, 256, 256 * 4])
# autoencoder = LinearAE([16, 8, 16])
# autoencoder = ConvAE(bottleneck=4)

epochs = 50
learning_rate = 0.001

criterion = PSNR
# criterion = MSELoss()

# for i in range(1, 3):
# autoencoder = LinearAE([256 * 4, 64 * i, 256 * 4])
autoencoder = ConvAE(bottleneck=4)
optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

trainer = CompTrainer(autoencoder, optimizer, data_loader)
trainer.train(epochs, criterion)

# autoencoder.to(device)

# for i in [16, 32]:
#     autoencoder = LinearAE([256 * 4, 64 * i, 256 * 4])
#     # autoencoder = ConvAE(bottleneck=16 * i)
#     optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

#     trainer = CompTrainer(autoencoder, optimizer, data_loader)
#     trainer.train(epochs, criterion)
