import torch

from .simple import SimpleLayerUnidirectional

class DropoutUnidirectional(SimpleLayerUnidirectional):

    def __init__(self, dropout: float | None):
        super().__init__(torch.nn.Dropout(dropout) if dropout else torch.nn.Identity())
