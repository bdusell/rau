import torch

from .stateless import StatelessLayerUnidirectional

class DropoutUnidirectional(StatelessLayerUnidirectional):

    def __init__(self, dropout: float | None):
        super().__init__(torch.nn.Dropout(dropout) if dropout else torch.nn.Identity())
