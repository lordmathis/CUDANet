import csv
from abc import ABC, abstractmethod

import numpy as np
import torch

_TORCH_TO_NUMPY: dict[torch.dtype, np.dtype] = {
    torch.float32: np.dtype("float32"),
    torch.float16: np.dtype("float16"),
    torch.int32:   np.dtype("int32"),
    torch.int64:   np.dtype("int64"),
}

class BaseGenerator(ABC):

    def __init__(self, seed, fixtures_path):
        self.fixtures_path = fixtures_path

        torch.manual_seed(seed)

    @abstractmethod
    def generate():
        ...

    @staticmethod
    def save_tensor(tensor: torch.Tensor, path: str) -> None:
        np_dtype = _TORCH_TO_NUMPY[tensor.dtype]
        tensor.detach().cpu().contiguous().numpy().astype(np_dtype, copy=False).tofile(path)

    @staticmethod
    def save_metadata(metadata: list[list[str]], path: str) -> None:
        with open(path, 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerows(metadata)