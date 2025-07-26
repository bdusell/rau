from typing import Optional

import torch

from rau.unidirectional import (
    Unidirectional,
    ResidualUnidirectional,
    StatelessLayerUnidirectional,
    DropoutUnidirectional
)

def get_unidirectional_sublayer(
    sublayer_func: Unidirectional,
    d_model: int,
    dropout: Optional[float]
) -> Unidirectional:
    return ResidualUnidirectional(
        StatelessLayerUnidirectional(torch.nn.LayerNorm(d_model)) |
        sublayer_func.main() |
        DropoutUnidirectional(dropout)
    )
