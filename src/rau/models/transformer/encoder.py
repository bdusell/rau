import torch

from rau.tools.torch.compose import Composable

from .common import add_tag
from .positional_encodings import SinusoidalPositionalEncodingCacher
from .input_layer import get_transformer_input_unidirectional

def get_transformer_encoder(
    vocabulary_size: int,
    shared_embeddings: torch.Tensor | None,
    positional_encoding_cacher: SinusoidalPositionalEncodingCacher | None,
    num_layers: int,
    d_model: int,
    num_heads: int,
    feedforward_size: int,
    dropout: float,
    use_padding: bool,
    tag: str | None = None
) -> torch.nn.Module:
    r"""Construct a bidirectional transformer
    :cite:p:`vaswani-etal-2017-attention` encoder.

    :param vocabulary_size: The size of the input vocabulary.
    :param shared_emeddings: An optional matrix of input embeddings that can be
        shared elsewhere.
    :param positional_encoding_cacher: Optional cache for computing positional
        encodings.
    :param num_layers: Number of layers.
    :param d_model: The size of the vector representations used in the model, or
        :math:`d_\mathrm{model}`.
    :param num_heads: Number of attention heads per layer.
    :param feedforward_size: Number of hidden units in each feedforward
        sublayer.
    :param dropout: Dropout rate used throughout the transformer.
    :param use_padding:
    :param tag:
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
        return self.layers(source_sequence, src_key_padding_mask=is_padding_mask)
