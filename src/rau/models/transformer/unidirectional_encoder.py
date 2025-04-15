from collections.abc import Callable, Iterable
from typing import Optional, Union

import torch

from rau.unidirectional import (
    Unidirectional,
    ForwardResult,
    OutputUnidirectional
)
from rau.models.common.shared_embeddings import get_shared_embeddings

from .common import add_tag
from .positional_encodings import SinusoidalPositionalEncodingCacher
from .input_layer import get_transformer_input_unidirectional
from .mask import make_causal_attention_mask

def get_unidirectional_transformer_encoder(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    tie_embeddings: bool,
    num_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    shared_embeddings: Optional[torch.Tensor]=None,
    positional_encoding_cacher: Optional[SinusoidalPositionalEncodingCacher]=None,
    tag: Optional[str]=None
) -> Unidirectional:
    if shared_embeddings is None:
        shared_embeddings = get_shared_embeddings(
            tie_embeddings,
            input_vocabulary_size,
            output_vocabulary_size,
            d_model,
            use_padding
        )
    return (
        get_transformer_input_unidirectional(
            vocabulary_size=input_vocabulary_size,
            d_model=d_model,
            dropout=dropout,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher
        ) |
        add_tag(UnidirectionalTransformerEncoderLayers(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_final_layer_norm=True
        ), tag) |
        OutputUnidirectional(
            input_size=d_model,
            vocabulary_size=output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            bias=False
        )
    )

class UnidirectionalTransformerEncoderLayers(Unidirectional):

    def __init__(self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
        use_final_layer_norm: bool
    ):
        super().__init__()
        self.layers = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(d_model) if use_final_layer_norm else None,
            enable_nested_tensor=False
        )
        self.d_model = d_model

    def forward(self,
        input_sequence: torch.Tensor,
        is_padding_mask: Optional[torch.Tensor]=None,
        initial_state: Optional[Unidirectional.State]=None,
        return_state: bool=False,
        include_first: bool=True
    ) -> Union[torch.Tensor, ForwardResult]:
        if initial_state is not None:
            # TODO
            raise NotImplementedError
        if return_state:
            # TODO
            raise NotImplementedError
        if include_first:
            raise ValueError('include_first must be False')
        return self.layers(
            src=input_sequence,
            mask=make_causal_attention_mask(
                sequence_length=input_sequence.size(1),
                device=input_sequence.device,
                dtype=input_sequence.dtype
            ),
            src_key_padding_mask=is_padding_mask
        )

    class State(Unidirectional.State):

        encoder: 'UnidirectionalTransformerEncoderLayers'
        previous_inputs: torch.Tensor

        def __init__(self,
            encoder: 'UnidirectionalTransformerEncoderLayers',
            previous_inputs: torch.Tensor
        ):
            super().__init__()
            self.encoder = encoder
            self.previous_inputs = previous_inputs

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return self.encoder.State(
                self.encoder,
                # Simply concatenate this input to the tensor of all previous
                # inputs.
                torch.concat([
                    self.previous_inputs,
                    input_tensor[:, None, :]
                ], dim=1)
            )

        def output(self) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
            # TODO This is very inefficient
            # NOTE This assumes there is no padding in the input
            full_output = self.encoder.forward(
                self.previous_inputs,
                include_first=False
            )
            return full_output[:, -1]

        def batch_size(self) -> int:
            return self.previous_inputs.size(0)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return self.encoder.State(
                self.encoder,
                func(self.previous_inputs)
            )

        # TODO Efficiently implement the methods below.

        def fastforward(self, input_sequence: torch.Tensor) -> Unidirectional.State:
            raise NotImplementedError

        def states(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Iterable[Unidirectional.State]:
            raise NotImplementedError

        def outputs(self,
            input_sequence: torch.Tensor,
            include_first: bool
        ) -> Union[Iterable[torch.Tensor], Iterable[tuple[torch.Tensor, ...]]]:
            raise NotImplementedError

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> Union[torch.Tensor, ForwardResult]:
            raise NotImplementedError

    def initial_state(self, batch_size: int) -> Unidirectional.State:
        tensor = next(self.parameters())
        return self.State(
            self,
            torch.empty(
                (batch_size, 0, self.d_model),
                dtype=tensor.dtype,
                device=tensor.device
            )
        )
