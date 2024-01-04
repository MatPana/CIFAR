from numpy import prod
from typing import List, Tuple
from torch import Tensor
from torch.nn import Module, Sequential, ReLU, Linear
from torch.nn.functional import sigmoid

class LinearAE(Module):
    def __init__(self, hidden: List[int], img_size: Tuple[int] = (3, 32, 32)) -> None:
        super().__init__()
        self.img_size = img_size
        self.name = f"Linear_{min(hidden)}"
        in_dim = prod(img_size)

        modules = []
        for hid in hidden:
            modules.append(Linear(in_dim, hid))
            modules.append(ReLU())
            in_dim = hid
        modules.append(Linear(in_dim, prod(img_size)))

        self.encoder_decoder = Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.encoder_decoder(x)
        x = sigmoid(x)
        x = x.view(-1, *self.img_size)
        return x
