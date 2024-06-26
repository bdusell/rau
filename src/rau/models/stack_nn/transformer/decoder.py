import functools

import torch

from rau.unidirectional import SimpleLayerUnidirectional, OutputUnidirectional
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.decoder import TransformerDecoderLayers

from .parse import get_stack_attention_func

def get_stack_transformer_decoder(
    input_vocabulary_size,
    output_vocabulary_size,
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
                yield TransformerDecoderLayers(
                    num_layers=num_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    use_final_layer_norm=False
                ).tag(layer_type)
            else:
                yield get_decoder_layer_with_custom_attention(
                    get_stack_attention_func(layer_type, layer_args, d_model),
                    d_model=d_model,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    num_cross_attention_heads=num_heads,
                    tag=layer_type,
                    cross_attention_tag='transformer'
                )
        yield SimpleLayerUnidirectional(torch.nn.LayerNorm(d_model))
        yield OutputUnidirectional(
            input_size=d_model,
            vocabulary_size=output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            bias=False
        )

    return functools.reduce(lambda x, y: x @ y, generate_layers())
