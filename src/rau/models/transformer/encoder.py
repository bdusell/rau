import torch

from rau.tools.torch.compose import Composable
from rau.models.common.add_tag import add_tag

from .positional_encodings import SinusoidalPositionalEncodingCacher
from .input_layer import get_transformer_input_unidirectional

def get_transformer_encoder(
    vocabulary_size: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    shared_embeddings: torch.Tensor | None,
    positional_encoding_cacher: SinusoidalPositionalEncodingCacher | None,
    tag: str | None = None
) -> torch.nn.Module:
    r"""Construct a bidirectional transformer encoder.

    It includes a scaled input embedding layer with sinusoidal positional
    encodings but *no* separate output layer. Its outputs are the outputs of
    layer norm applied to the outputs of the last layer.

    :param vocabulary_size: The size of the input vocabulary.
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
        :py:class:`TransformerEncoderLayers` for argument routing.
    :return: A module. Unless ``tag`` is given, it accepts the same arguments as
        :py:class:`TransformerEncoderLayers`.
    """
    return (
        Composable(
            get_transformer_input_unidirectional(
                vocabulary_size,
                d_model,
                dropout,
                use_padding,
                shared_embeddings,
                positional_encoding_cacher
            )
        ).kwargs(include_first=False) |
        add_tag(Composable(
            TransformerEncoderLayers(
                num_layers,
                d_model,
                num_heads,
                feedforward_size,
                dropout,
                use_final_layer_norm=True
            )
        ), tag)
    )

class TransformerEncoderLayers(torch.nn.Module):
    r"""A bidirectional transformer encoder without input or outpt layers."""

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
            # This is not compatible with norm_first=True.
            enable_nested_tensor=False
        )

    def forward(self,
        source_sequence: torch.Tensor,
        is_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        :param is_padding_mask: A bool tensor indicating which positions in the
            input should be treated as padding symbols and ignored. This always
            needs to be used if you are using a minibatch with sequences of
            different lengths. Its size should be
            :math:`\text{batch size} \times \text{input length}`. A value of
            true indicates that a token is padding.
        """
        return self.layers(source_sequence, src_key_padding_mask=is_padding_mask)
