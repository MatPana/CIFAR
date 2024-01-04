from typing import Optional
import numpy as np
from torch import tensor, float32
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import inject_noise, unpickle

class DataPrep:
    def  __init__(self, dir: str) -> None:
        self.dir = dir

    def get_dataloaders(self, batch_size: int = 128) -> DataLoader:
        data = self._load_arrays()
        dataset = TensorDataset(tensor(data, dtype=float32)) 
        loader = DataLoader(dataset, batch_size=batch_size)
        return loader

    def _load_arrays(self) -> np.array:
        data = unpickle(self.dir)
        data = data[b"data"]
        data = data.reshape(-1, 3, 32, 32)
        data = data.astype(np.float32) / 255
        return data
