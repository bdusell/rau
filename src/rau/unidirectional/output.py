from typing import Optional

import torch

from rau.tools.torch.tied_linear import get_linear
from .simple import SimpleLayerUnidirectional

class OutputUnidirectional(SimpleLayerUnidirectional):

    def __init__(self,
        input_size: int,
        vocabulary_size: int,
        shared_embeddings: Optional[torch.Tensor]=None,
        bias: bool=True
    ):
        super().__init__(get_linear(
            input_size,
            vocabulary_size,
            shared_embeddings,
            bias
        ))
