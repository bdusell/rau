import torch

from rau.unidirectional import Unidirectional
from .encoder import get_transformer_encoder
from .decoder import get_transformer_decoder
from .positional_encodings import SinusoidalPositionalEncodingCacher

def get_transformer_encoder_decoder(
    source_vocabulary_size: int,
    target_input_vocabulary_size: int,
    target_output_vocabulary_size: int,
    tie_embeddings: bool,
    num_encoder_layers: int,
    num_decoder_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_source_padding: bool = True,
    use_target_padding: bool = True
) -> 'TransformerEncoderDecoder':
    r"""Construct a transformer encoder-decoder.

    It includes a scaled input embedding layers with sinusoidal positional
    encodings in the encoder and decoder and an output layer in the decoder for
    predicting logits.

    :param source_vocabulary_size: The size of the vocabulary used by the
        encoder.
    :param target_input_vocabulary_size: The size of the input vocabulary used
        by the decoder.
    :param target_output_vocabulary_size: The size of the output vocabulary used
        by the decoder.
    :param tie_embeddings: Whether to tie the input and output embeddings.
    :param num_encoder_layers: Number of layers to use in the encoder.
    :param num_decoder_layers: Number of layers to use in the decoder.
    :param d_model: The size of the vector representations used in the model, or
        :math:`d_\mathrm{model}`.
    :param num_heads: Number of attention heads per layer.
    :param feedforward_size: Number of hidden units in each feedforward
        sublayer.
    :param dropout: Dropout rate used throughout the transformer.
    :param use_source_padding: Whether to ensure that the embedding matrix for
        the encoder is big enough to accommodate an index for a reserved padding
        symbol.
    :param use_target_padding: Whether to ensure that the embedding matrix for
        the decoder is big enough to accommodate an index for a reserved padding
        symbol.
    """
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
    # NOTE It's ok to simply pass the same parameter to multiple sub-modules.
    # https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
    return TransformerEncoderDecoder(
        get_transformer_encoder(
            vocabulary_size=source_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_source_padding
        ),
        get_transformer_decoder(
            input_vocabulary_size=target_input_vocabulary_size,
            output_vocabulary_size=target_output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_target_padding
        )
    )

def get_shared_embeddings(
    tie_embeddings: bool,
    source_vocabulary_size: int,
    target_input_vocabulary_size: int,
    target_output_vocabulary_size: int,
    d_model: int,
    use_source_padding: bool,
    use_target_padding: bool
) -> torch.Tensor | None:
    if tie_embeddings:
        return construct_shared_embeddings(
            source_vocabulary_size,
            target_input_vocabulary_size,
            target_output_vocabulary_size,
            d_model,
            use_source_padding,
            use_target_padding
        )
    else:
        return None

def construct_shared_embeddings(
    source_vocabulary_size,
    target_input_vocabulary_size,
    target_output_vocabulary_size,
    d_model,
    use_source_padding,
    use_target_padding
):
    if target_output_vocabulary_size > target_input_vocabulary_size:
        raise ValueError(
            f'target output vocabuary size ({target_output_vocabulary_size}) '
            f'must be greater than target input vocabulary size '
            f'({target_input_vocabulary_size})')
    vocab_size = max(
        source_vocabulary_size + int(use_source_padding),
        target_input_vocabulary_size + int(use_target_padding)
    )
    return torch.nn.Parameter(torch.zeros(vocab_size, d_model))

class TransformerEncoderDecoder(torch.nn.Module):
    r"""A transformer encoder-decoder."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
        source_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        source_is_padding_mask: torch.Tensor | None = None,
        target_is_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        :param source_sequence: A batch of source sequences. A tensor of ints of
            size :math:`\text{batch size} \times \text{source length}`.
        :param target_sequence: A batch of target sequences. A tensor of ints of
            size :math:`\text{batch size} \times \text{target length}`.
        :param source_is_padding_mask: Bool tensor indicating which positions in
            the source correspond to padding symbols that should be ignored.
            This always needs to be used if you are using a minibatch with
            source sequences of different lengths. Its size should be
            :math:`\text{batch size} \times \text{source length}`. A value of
            true indicates that a token is padding.
        :param target_is_padding_mask: Optional bool tensor indicating which
            positions in the target correspond to padding symbols that should be
            ignored. Since the decoder is already causally masked, this should
            usually not be necessary, and it is better not to use it. Its size
            should be :math:`\text{batch size} \times \text{target length}`. A
            value of true indicates that a token is padding.
        :return: The output logits of the decoder, of size :math:`\text{batch
            size} \times \text{target length} \times \text{target vocabulary
            size}`.
        """
        encoder_outputs = self.encoder(
            source_sequence,
            is_padding_mask=source_is_padding_mask
        )
        return self.decoder(
            target_sequence,
            encoder_sequence=encoder_outputs,
            input_is_padding_mask=target_is_padding_mask,
            encoder_is_padding_mask=source_is_padding_mask,
            include_first=False
        )

    def initial_decoder_state(self,
        source_sequence: torch.Tensor,
        source_is_padding_mask: torch.Tensor | None
    ) -> Unidirectional.State:
        r"""Given a batch of source sequences, compute the initial state of the
        decoder.

        :param source_sequence: A batch of source sequences. A tensor of ints of
            size :math:`\text{batch size} \times \text{source length}`.
        :param source_is_padding_mask: Bool tensor indicating which positions
            in the source correspond to padding symbols that should be ignored.
            This always needs to be used if you are using a minibatch with
            source sequences of different lengths. Its size should be
            :math:`\text{batch size} \times \text{source length}`. A value of
            true indicates that a token is padding.
        :return: The initial state of the decoder, conditioned on the source
            sequences.
        """
        encoder_outputs = self.encoder(
            source_sequence,
            is_padding_mask=source_is_padding_mask
        )
        return self.decoder.initial_state(
            batch_size=encoder_outputs.size(0),
            encoder_sequence=encoder_outputs,
            encoder_is_padding_mask=source_is_padding_mask
        )
