from typing import Any

import torch

from rau.unidirectional import Unidirectional
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.encoder_decoder import get_shared_embeddings

from .parse import StackTransformerLayers
from .encoder import get_stack_transformer_encoder
from .decoder import get_stack_transformer_decoder

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
