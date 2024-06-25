import functools

import torch

from rau.unidirectional import (
    Unidirectional,
    SimpleLayerUnidirectional,
    OutputUnidirectional
)
from rau.models.common.shared_embeddings import get_shared_embeddings
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.unidirectional_encoder import UnidirectionalTransformerEncoderLayers
from .parse import StackTransformerLayers

def get_unidirectional_stack_transformer_encoder(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    tie_embeddings: bool,
    layers: StackTransformerLayers,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    shared_embeddings: torch.Tensor | None=None,
    positional_encoding_cacher: SinusoidalPositionalEncodingCacher | None=None,
    tag: str | None=None
) -> Unidirectional:

    if shared_embeddings is None:
        shared_embeddings = get_shared_embeddings(
            tie_embeddings,
            input_vocabulary_size,
            output_vocabulary_size,
            d_model,
            use_padding
        )

    def generate_layers():
        yield get_transformer_input_unidirectional(
            vocabulary_size=input_vocabulary_size,
            d_model=d_model,
            dropout=dropout,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher
        )
        for layer_type, layer_args in layers:
            if layer_type == 'transformer':
                num_layers, = layer_args
                yield UnidirectionalTransformerEncoderLayers(
                    num_layers=num_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    use_final_layer_norm=False
                )
            else:
                yield get_unidirectional_encoder_layer_with_custom_attention(
                    get_stack_attention_func(layer_type, layer_args, d_model),
                    d_model=d_model,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    tag=layer_type
                )
        yield SimpleLayerUnidirectional(torch.nn.LayerNorm(d_model))
        yield OutputUnidirectional(
            input_size=d_model,
            vocabulary_size=output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            bias=False
        )

    return functools.reduce(lambda x, y: x @ y, generate_layers())
