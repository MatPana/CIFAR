import csv
from os.path import join
from typing import List
from torch import save
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.utils import PSNR, inject_noise


class NoiseTrainer:
    def __init__(
        self,
        model: Module,
        optimiser: Optimizer,
        data_loader: DataLoader,
        noise: float = 0.1,
    ) -> None:
        self.model = model
        self.optimiser = optimiser
        self.data_loader = data_loader
        self.noise = noise

    def train(
        self,
        epochs: int = 1000,
        criterion: callable = PSNR,
        result_dir: str = "results",
    ) -> None:
        baseline_loss = self.get_benchmark()
        print("Baseline loss: " + str(baseline_loss))
        losses = [baseline_loss]
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch,) in self.data_loader:
                inputs = inject_noise(batch, self.noise)
                # inputs = inputs.to(device)
                self.optimiser.zero_grad()

                outputs = self.model(inputs)

                loss = criterion(batch, outputs)

                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
            losses += [epoch_loss]
            save(self.model.state_dict(), f"./{result_dir}/autoencoder_{epoch}.pt")
        self.log(losses, result_dir)

    def get_benchmark(self, criterion: callable = PSNR) -> float:
        # Generation of baseline for noise.
        running_loss = 0.0
        for (batch,) in self.data_loader:
            inputs = inject_noise(batch, self.noise)
            loss = criterion(batch, inputs)
            running_loss += loss.item()

        epoch_loss = running_loss / len(self.data_loader)

        return epoch_loss

    def log(self, losses: List, result_dir: str) -> None:
        write_dir = result_dir + "/" + self.model.name + "_" + str(self.noise) + ".csv"

        with open(write_dir, "w") as f:
            write = csv.writer(f)
            write.writerow(losses)
