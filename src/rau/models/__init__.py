r"""This module contains implementations of neural network architectures."""

from .transformer.unidirectional_encoder import (
    get_unidirectional_transformer_encoder,
    UnidirectionalTransformerEncoderLayers
)
from .transformer.encoder import (
    get_transformer_encoder,
    TransformerEncoderLayers
)
from .transformer.decoder import (
    get_transformer_decoder,
    TransformerDecoderLayers
)
from .transformer.encoder_decoder import (
    get_transformer_encoder_decoder,
    TransformerEncoderDecoder
)
from .transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from .rnn.simple_rnn import SimpleRNN
from .rnn.lstm import LSTM
from .rnn.language_model import (
    get_simple_rnn_language_model,
    get_lstm_language_model,
    get_rnn_language_model
)
from .common.shared_embeddings import get_shared_embeddings
