import pickle
from torch import tensor, log10, randn, clamp


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def PSNR(inputs: tensor, outputs: tensor) -> tensor:
    maxes = inputs.view(inputs.size(0), -1).max(dim=1).values
    squared_diff = (inputs - outputs) ** 2
    mse_per_image = squared_diff.view(squared_diff.size(0), -1).mean(dim=1)
    psnr = 20 * log10(maxes) - 10 * log10(mse_per_image)
    return -psnr.mean()

def inject_noise(inputs: tensor, noise_std: float = 0.1) -> tensor:
    shape = inputs.shape
    noise_tensor = randn(shape) * noise_std
    inputs = inputs + noise_tensor
    inputs = clamp(inputs, 0, 1)
    return inputs
