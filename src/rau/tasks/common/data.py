import pathlib

import torch

def load_prepared_data_file(path: pathlib.Path) -> list[torch.Tensor]:
    return [torch.tensor(x) for x in torch.load(path)]