r"""This module contains implementations of neural network architectures.

Notes on the Transformer Architecture
-------------------------------------

The following notes apply to all instances of the transformer architecture
:cite:p:`vaswani-etal-2017-attention`.

* They use pre-norm instead of post-norm
  :cite:p:`wang-etal-2019-learning,nguyen-salazar-2019-transformers`.
* Dropout is applied  to the same places as in
  :cite:t:`vaswani-etal-2017-attention` and also to the hidden units of
  feedforward sublayers and the attention probabilities of the attention
  mechanism.
* They use the sinusoidal positional encodings as originally proposed by
  :cite:t:`vaswani-etal-2017-attention`.
"""

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
