import torch

from rau.unidirectional import SimpleReshapingLayerUnidirectional

class CrossAttention(torch.nn.Module):

    def __init__(self,
        d_model: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self,
        input_sequence: torch.Tensor,
        encoder_sequence: torch.Tensor,
        encoder_is_padding_mask: torch.Tensor | None=None
    ) -> torch.Tensor:
        """
        :param input_sequence: The target sequence that is given as input to
            the cross-attention sublayer.
        :param encoder_sequence: The output sequence of the encoder.
        """
        return self.attention(
            input_sequence,
            encoder_sequence,
            encoder_sequence,
            key_padding_mask=encoder_is_padding_mask,
            need_weights=False
        )[0]

class CrossAttentionUnidirectional(SimpleReshapingLayerUnidirectional):

    def __init__(self,
        d_model: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__(CrossAttention(d_model, num_heads, dropout))

    def transform_kwargs(self, kwargs, func):
        kwargs = kwargs.copy()
        kwargs['encoder_sequence'] = func(kwargs['encoder_sequence'])
        kwargs['encoder_is_padding_mask'] = func(kwargs['encoder_is_padding_mask'])
        return kwargs
