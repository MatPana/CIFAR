import csv
from os.path import join
from typing import List
from torch import save
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.utils import PSNR


class CompTrainer:
    def __init__(
        self, model: Module, optimiser: Optimizer, data_loader: DataLoader
    ) -> None:
        self.model = model
        self.optimiser = optimiser
        self.data_loader = data_loader

    def train(
        self,
        epochs: int = 1000,
        criterion: callable = PSNR,
        result_dir: str = "results",
    ) -> None:
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch,) in self.data_loader:
                inputs = batch
                # inputs = inputs.to(device)
                self.optimiser.zero_grad()

                outputs = self.model(inputs)

                loss = criterion(outputs, inputs)

                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
            losses += [epoch_loss]
            save(
                self.model.state_dict(),
                f"./{result_dir}/autoencoder_{self.model.name}_{epoch}.pt",
            )
        self.log(losses, result_dir)

    def log(self, losses: List, result_dir: str) -> None:
        write_dir = result_dir + "/" + self.model.name + ".csv"

        with open(write_dir, "w") as f:
            write = csv.writer(f)
            write.writerow(losses)
