from rau.unidirectional import Unidirectional
from rau.models.rnn.language_model import get_rnn_language_model
from rau.models.rnn.simple_rnn import SimpleRNN
from rau.models.rnn.lstm import LSTM
from rau.models.common.add_tag import add_tag

from .parse import StackRNNController, StackRNNStack
from .stratification import StratificationStackRNN
from .superposition import SuperpositionStackRNN
from .nondeterministic import NondeterministicStackRNN
#from .vector_nondeterministic import VectorNondeterministicStackRNN

def get_stack_rnn_language_model(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    hidden_units: int,
    layers: int,
    controller: StackRNNController,
    stack: StackRNNStack,
    dropout: float,
    learned_hidden_state: bool,
    use_padding: bool,
    tag: str | None=None
) -> Unidirectional:
    return get_rnn_language_model(
        add_tag(get_stack_rnn_recurrence(
            controller=controller,
            stack=stack,
            input_size=hidden_units,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            learned_hidden_state=learned_hidden_state
        ), tag),
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        hidden_units=hidden_units,
        dropout=dropout,
        use_padding=use_padding
    )

def get_stack_rnn_recurrence(
    controller: StackRNNController,
    stack: StackRNNStack,
    input_size: int,
    hidden_units: int,
    layers: int,
    dropout: float,
    learned_hidden_state: bool
) -> Unidirectional:

    if controller == 'rnn':
        def controller(input_size):
            return SimpleRNN(
                input_size=input_size,
                hidden_units=hidden_units,
                layers=layers,
                dropout=dropout,
                learned_hidden_state=learned_hidden_state
            )
    elif controller == 'lstm':
        def controller(input_size):
            return LSTM(
                input_size=input_size,
                hidden_units=hidden_units,
                layers=layers,
                dropout=dropout,
                learned_hidden_state=learned_hidden_state
            )
    else:
        raise ValueError
    controller_output_size = hidden_units
    include_reading_in_output = False
    reading_layer_sizes = None

    stack_name, stack_args = stack
    if stack_name == 'stratification':
        stack_embedding_size, = stack_args
        return StratificationStackRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
    elif stack_name == 'superposition':
        stack_embedding_size, = stack_args
        return SuperpositionStackRNN(
            input_size=input_size,
            stack_embedding_size=stack_embedding_size,
            push_hidden_state=False,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
    elif stack_name == 'nondeterministic':
        num_states, stack_alphabet_size = stack_args
        return NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
    elif stack_name == 'vector-nondeterministic':
        num_states, stack_alphabet_size, stack_embedding_size = stack_args
        return VectorNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
    else:
        raise ValueError
