import functools
from typing import Any

import torch

from rau.unidirectional import (
    Unidirectional,
    SimpleLayerUnidirectional,
    OutputUnidirectional
)
from rau.tools.torch.compose import Composable
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.encoder import TransformerEncoderLayers
from rau.models.transformer.decoder import TransformerDecoderLayers
from rau.models.transformer.encoder_decoder import get_shared_embeddings

from .parse import (
    StackTransformerLayers,
    get_stack_attention_func
)
from .unidirectional_encoder import (
    get_unidirectional_encoder_layer_with_custom_attention
)

class EncoderDecoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
        source_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        encoder_kwargs: dict[str, Any],
        decoder_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        self._run_encoder(source_sequence, encoder_kwargs, decoder_kwargs)
        return self.decoder(target_sequence, **decoder_kwargs, include_first=False)

    def initial_decoder_state(self,
        source_sequence: torch.Tensor,
        encoder_kwargs: dict[str, Any],
        decoder_kwargs: dict[str, Any]
    ) -> Unidirectional.State:
        encoder_outputs = self._run_encoder(source_sequence, encoder_kwargs, decoder_kwargs)
        return self.decoder.initial_state(
            batch_size=encoder_outputs.size(0),
            **decoder_kwargs
        )

    def _run_encoder(self, source_sequence, encoder_kwargs, decoder_kwargs):
        # Run the encoder and add its output as a keyword argument to the
        # decoder.
        encoder_outputs = self.encoder(source_sequence, **encoder_kwargs)
        decoder_kwargs['tag_kwargs']['transformer']['encoder_sequence'] = encoder_outputs
        return encoder_outputs

def get_stack_transformer_encoder_decoder(
    source_vocabulary_size: int,
    target_input_vocabulary_size: int,
    target_output_vocabulary_size: int,
    tie_embeddings: bool,
    encoder_layers: StackTransformerLayers,
    decoder_layers: StackTransformerLayers,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float | None,
    use_source_padding: bool,
    use_target_padding: bool
) -> EncoderDecoder:
    shared_embeddings = get_shared_embeddings(
        tie_embeddings,
        source_vocabulary_size,
        target_input_vocabulary_size,
        target_output_vocabulary_size,
        d_model,
        use_source_padding,
        use_target_padding
    )
    positional_encoding_cacher = SinusoidalPositionalEncodingCacher()
    return EncoderDecoder(
        get_stack_transformer_encoder(
            vocabulary_size=source_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            layers=encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_source_padding
        ),
        get_stack_transformer_decoder(
            input_vocabulary_size=target_input_vocabulary_size,
            output_vocabulary_size=target_output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            layers=decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_target_padding
        )
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
