import functools

import torch

from rau.tools.torch.compose import Composable
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.encoder import TransformerEncoderLayers

from .parse import get_stack_attention_func
from .unidirectional_encoder import (
    get_unidirectional_encoder_layer_with_custom_attention
)

def get_stack_transformer_encoder(
    vocabulary_size,
    shared_embeddings,
    positional_encoding_cacher,
    layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_padding
):

    def generate_layers():
        yield Composable(get_transformer_input_unidirectional(
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            dropout=dropout,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher
        )).kwargs(include_first=False)
        for layer_type, layer_args in layers:
            if layer_type == 'transformer':
                num_layers, = layer_args
                yield Composable(TransformerEncoderLayers(
                    num_layers=num_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    use_final_layer_norm=False
                )).tag(layer_type)
            else:
                yield Composable(get_unidirectional_encoder_layer_with_custom_attention(
                    get_stack_attention_func(layer_type, layer_args, d_model),
                    d_model=d_model,
                    feedforward_size=feedforward_size,
                    dropout=dropout
                )).kwargs(include_first=False).tag(layer_type)
        yield Composable(torch.nn.LayerNorm(d_model))

    return functools.reduce(lambda x, y: x @ y, generate_layers())
