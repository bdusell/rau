from collections.abc import Callable, Iterable

import torch

from rau.unidirectional import (
    Unidirectional,
    ForwardResult,
    OutputUnidirectional
)
from rau.models.common.add_tag import add_tag

from .positional_encodings import SinusoidalPositionalEncodingCacher
from .input_layer import get_transformer_input_unidirectional
from .mask import make_causal_attention_mask

def get_transformer_decoder(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    shared_embeddings: torch.Tensor | None,
    positional_encoding_cacher: SinusoidalPositionalEncodingCacher | None,
    tag: str | None = None
) -> Unidirectional:
    r"""Construct a transformer decoder with cross-attention.

    It includes a scaled input embedding layer with sinusoidal positional
    encodings and an output layer for predicting logits.

    :param input_vocabulary_size: The size of the input vocabulary.
    :param output_vocabulary_size: The size of the output vocabulary.
    :param num_layers: Number of layers.
    :param d_model: The size of the vector representations used in the model, or
        :math:`d_\mathrm{model}`.
    :param num_heads: Number of attention heads per layer.
    :param feedforward_size: Number of hidden units in each feedforward
        sublayer.
    :param dropout: Dropout rate used throughout the transformer.
    :param use_padding: Whether to ensure that the embedding matrix is big
        enough to accommodate an index for a reserved padding symbol.
    :param shared_embeddings: An optional matrix of embeddings that can be
        shared elsewhere.
    :param positional_encoding_cacher: Optional cache for computing positional
        encodings that can be shared elsewhere.
    :param tag: An optional tag to add to the inner
        :py:class:`UnidirectionalTransformerEncoderLayers` for argument routing.
    :return: A module. Unless ``tag`` is given, it accepts the same arguments as
        :py:class:`TransformerDecoderLayers`.
    """
    return (
        get_transformer_input_unidirectional(
            vocabulary_size=input_vocabulary_size,
            d_model=d_model,
            dropout=dropout,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher
        ) |
        add_tag(TransformerDecoderLayers(
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

class TransformerDecoderLayers(Unidirectional):
    r"""A transformer decoder without input or output layers."""

    def __init__(self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
        use_final_layer_norm: bool
    ):
        super().__init__()
        self.layers = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(d_model) if use_final_layer_norm else None
        )

    def forward(self,
        input_sequence: torch.Tensor,
        encoder_sequence: torch.Tensor,
        input_is_padding_mask: torch.Tensor | None = None,
        encoder_is_padding_mask: torch.Tensor | None = None,
        initial_state: Unidirectional.State | None = None,
        return_state: bool = False,
        include_first: bool = True
    ) -> torch.Tensor | ForwardResult:
        r"""
        :param encoder_sequence: The output sequence of the encoder.
        :param input_is_padding_mask: Optional bool tensor indicating which
            positions in the decoder input correspond to padding symbols that
            should be ignored. Since the decoder is already causally masked,
            this should usually not be necessary, and it is better not to use
            it. Its size should be :math:`\text{batch size} \times \text{decoder
            input length}`. A value of true indicates that a token is padding.
        :param encoder_is_padding_mask: Bool tensor indicating which positions
            in the encoder input correspond to padding symbols that should be
            ignored. This always needs to be used if you are using a minibatch
            with input sequences of different lengths. Its size should be
            :math:`\text{batch size} \times \text{encoder input length}`. A
            value of true indicates that a token is padding.
        """
        if initial_state is not None:
            # TODO
            raise NotImplementedError
        if return_state:
            # TODO
            raise NotImplementedError
        if include_first:
            raise ValueError('include_first must be False')
        return self.layers(
            tgt=input_sequence,
            memory=encoder_sequence,
            tgt_mask=make_causal_attention_mask(
                sequence_length=input_sequence.size(1),
                device=input_sequence.device,
                dtype=input_sequence.dtype
            ),
            tgt_key_padding_mask=input_is_padding_mask,
            memory_key_padding_mask=encoder_is_padding_mask
        )

    class State(Unidirectional.State):

        decoder: 'TransformerDecoderLayers'
        encoder_sequence: torch.Tensor
        encoder_is_padding_mask: torch.Tensor
        previous_inputs: torch.Tensor

        def __init__(self,
            decoder: 'TransformerDecoderLayers',
            encoder_sequence: torch.Tensor,
            encoder_is_padding_mask: torch.Tensor,
            previous_inputs: torch.Tensor
        ):
            super().__init__()
            self.decoder = decoder
            self.encoder_sequence = encoder_sequence
            self.encoder_is_padding_mask = encoder_is_padding_mask
            self.previous_inputs = previous_inputs

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return TransformerDecoderLayers.State(
                self.decoder,
                self.encoder_sequence,
                self.encoder_is_padding_mask,
                # Simply concatenate this input to the tensor of all previous
                # inputs.
                torch.concat([
                    self.previous_inputs,
                    input_tensor[:, None, :]
                ], dim=1)
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
            # TODO This is very inefficient
            # NOTE This assumes there is no padding in the decoder input
            full_output = self.decoder.forward(
                self.previous_inputs,
                self.encoder_sequence,
                encoder_is_padding_mask=self.encoder_is_padding_mask,
                include_first=False
            )
            return full_output[:, -1]

        def batch_size(self) -> int:
            return self.previous_inputs.size(0)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return TransformerDecoderLayers.State(
                self.decoder,
                func(self.encoder_sequence),
                func(self.encoder_is_padding_mask),
                func(self.previous_inputs)
            )

        # TODO

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
        ) -> Iterable[torch.Tensor] | Iterable[tuple[torch.Tensor, ...]]:
            raise NotImplementedError

        def forward(self,
            input_sequence: torch.Tensor,
            return_state: bool,
            include_first: bool
        ) -> torch.Tensor | ForwardResult:
            raise NotImplementedError

    def initial_state(self,
        batch_size: int,
        encoder_sequence: torch.Tensor,
        encoder_is_padding_mask: torch.Tensor
    ) -> Unidirectional.State:
        return self.State(
            self,
            encoder_sequence,
            encoder_is_padding_mask,
            torch.empty(
                (batch_size, 0, encoder_sequence.size(2)),
                dtype=encoder_sequence.dtype,
                device=encoder_sequence.device
            )
        )
