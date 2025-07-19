from typing import Literal

from rau.unidirectional import (
    Unidirectional,
    EmbeddingUnidirectional,
    DropoutUnidirectional,
    OutputUnidirectional
)
from rau.models.common.shared_embeddings import get_shared_embeddings

from .simple_rnn import SimpleRNN
from .lstm import LSTM

def get_simple_rnn_language_model(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    hidden_units: int,
    layers: int = 1,
    dropout: float = 0,
    nonlinearity: Literal['tanh', 'relu'] = 'tanh',
    bias: bool = True,
    learned_hidden_state: bool = True,
    use_extra_bias: bool = False,
    use_padding: bool = False
) -> Unidirectional:
    r"""Construct a simple RNN language model.

    The embedding size and input size are assumed to be the same as the hidden
    size.

    The input and output embeddings are tied.

    :param input_vocabulary_size: The size of the input vocabulary.
    :param output_vocabulary_size: The size of the output vocabulary.
    :param hidden_units: The number of hidden units in each layer.
    :param layers: The number of layers in the RNN.
    :param dropout: The amount of dropout applied to inputs, in between layers,
        and the last layer outputs.
    :param nonlinearity: The non-linearity applied to hidden units. Either
        ``'tanh'`` or ``'relu'``.
    :param bias: Whether to use bias terms.
    :param learned_hidden_state: Whether the initial hidden state should be a
        learned parameter. If true, the initial hidden state will be the result
        of passing learned parameters through the activation function. If false,
        the initial state will be zeros.
    :param use_extra_bias: The built-in PyTorch implementation of the RNN
        includes redundant bias terms, resulting in more parameters than
        necessary. If this is true, the extra bias terms are kept. Otherwise,
        they are removed.
    :param use_padding: Whether to ensure that the embedding matrix is big
        enough to accommodate an index for a reserved padding symbol.
    :return: A simple RNN language model with input and output layers. It
        accepts the same arguments as :py:class:`~rau.models.SimpleRNN`.
    """
    return get_rnn_language_model(
        SimpleRNN(
            input_size=hidden_units,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            nonlinearity=nonlinearity,
            bias=bias,
            learned_hidden_state=learned_hidden_state,
            use_extra_bias=use_extra_bias
        ),
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        hidden_units=hidden_units,
        dropout=dropout,
        use_padding=use_padding
    )

def get_lstm_language_model(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    hidden_units: int,
    layers: int = 1,
    dropout: float = 0,
    bias: bool = True,
    learned_hidden_state: bool = True,
    use_extra_bias: bool = False,
    use_padding: bool = False
) -> Unidirectional:
    r"""Construct an LSTM language model.

    The embedding size and input size are assumed to be the same as the hidden
    size.

    The input and output embeddings are tied.

    :param input_vocabulary_size: The size of the input vocabulary.
    :param output_vocabulary_size: The size of the output vocabulary.
    :param hidden_units: The number of hidden units in each layer.
    :param layers: The number of layers in the RNN.
    :param dropout: The amount of dropout applied to inputs, in between layers,
        and the last layer outputs.
    :param bias: Whether to use bias terms.
    :param learned_hidden_state: Whether the initial hidden state should be a
        learned parameter. If true, the initial hidden state will be the result
        of passing learned parameters through the activation function. If false,
        the initial state will be zeros.
    :param use_extra_bias: The built-in PyTorch implementation of the RNN
        includes redundant bias terms, resulting in more parameters than
        necessary. If this is true, the extra bias terms are kept. Otherwise,
        they are removed.
    :param use_padding: Whether to ensure that the embedding matrix is big
        enough to accommodate an index for a reserved padding symbol.
    :return: An LSTM language model with input and output layers. It accepts the
        same arguments as :py:class:`~rau.models.LSTM`.
    """
    return get_rnn_language_model(
        LSTM(
            input_size=hidden_units,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            bias=bias,
            learned_hidden_state=learned_hidden_state,
            use_extra_bias=use_extra_bias
        ),
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        hidden_units=hidden_units,
        dropout=dropout,
        use_padding=use_padding
    )

def get_rnn_language_model(
    recurrence: Unidirectional,
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    hidden_units: int,
    dropout: float = 0,
    use_padding: bool = False
) -> Unidirectional:
    r"""Wrap any recurrent network with input and output layers to make it a
    language model.

    :param recurrence: A unidirectional representing the recurrent part of the
        language model that will be wrapped with input and output layers.
    :param input_vocabulary_size: The size of the input vocabulary.
    :param output_vocabulary_size: The size of the output vocabulary.
    :param hidden_units: The size of the output vectors from ``recurrence``.
    :param dropout: The dropout rate applied to the input embeddings and to the
        hidden states before the output layer.
    :param use_padding: Whether to ensure that the embedding matrix is big
        enough to accommodate an index for a reserved padding symbol.
    :return: A module that accepts the same arguments as ``recurrence``.
    """
    shared_embeddings = get_shared_embeddings(
        tie_embeddings=True,
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        embedding_size=hidden_units,
        use_padding=use_padding
    )
    return (
        EmbeddingUnidirectional(
            vocabulary_size=input_vocabulary_size,
            output_size=hidden_units,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings
        ) |
        DropoutUnidirectional(dropout) |
        recurrence.main() |
        DropoutUnidirectional(dropout) |
        OutputUnidirectional(
            input_size=hidden_units,
            vocabulary_size=output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            bias=False
        )
    )
