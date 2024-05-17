import pathlib
from typing import Any, Generic, TypeVar

import torch

def load_prepared_data_file(path: pathlib.Path) -> list[torch.Tensor]:
    return [torch.tensor(x) for x in torch.load(path)]

Example = TypeVar('Example')

# TODO Is this not used?
class Dataset(Generic[Example]):
    training_data: list[Example]
    validation_data: list[Example]
