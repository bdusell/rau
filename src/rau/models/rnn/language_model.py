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
    layers: int,
    dropout: float,
    learned_hidden_state: bool,
    use_padding: bool
) -> Unidirectional:
    return get_rnn_language_model(
        SimpleRNN(
            input_size=hidden_units,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            learned_hidden_state=True
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
    layers: int,
    dropout: float,
    learned_hidden_state: bool,
    use_padding: bool
) -> Unidirectional:
    return get_rnn_language_model(
        LSTM(
            input_size=hidden_units,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            learned_hidden_state=True
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
    dropout: float,
    use_padding: bool
):
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
        ) @
        DropoutUnidirectional(dropout) @
        recurrence.main() @
        DropoutUnidirectional(dropout) @
        OutputUnidirectional(
            input_size=hidden_units,
            vocabulary_size=output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            bias=False
        )
    )
