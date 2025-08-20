from rau.unidirectional import Unidirectional
from rau.models.rnn.language_model import get_rnn_language_model
from rau.models.rnn.simple_rnn import SimpleRNN
from rau.models.rnn.lstm import LSTM
from rau.models.common.add_tag import add_tag

from .parse import StackRNNController, StackRNNStack
from .stratification import StratificationStackRNN
from .superposition import SuperpositionStackRNN
from .nondeterministic import NondeterministicStackRNN
from .vector_nondeterministic import VectorNondeterministicStackRNN

def get_stack_rnn_language_model(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    controller: StackRNNController,
    stack: StackRNNStack,
    hidden_units: int,
    layers: int = 1,
    dropout: float = 0,
    learned_initial_state: bool = True,
    include_reading_in_output: bool = False,
    use_padding: bool = False,
    tag: str | None = None
) -> Unidirectional:
    recurrence = get_stack_rnn_recurrence(
        controller=controller,
        stack=stack,
        hidden_units=hidden_units,
        layers=layers,
        dropout=dropout,
        learned_initial_state=learned_initial_state,
        include_reading_in_output=include_reading_in_output
    )
    return get_rnn_language_model(
        add_tag(recurrence, tag),
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        hidden_units=recurrence.output_size(),
        dropout=dropout,
        use_padding=use_padding
    )

def get_stack_rnn_recurrence(
    controller: StackRNNController,
    stack: StackRNNStack,
    hidden_units: int,
    layers: int,
    dropout: float,
    learned_initial_state: bool,
    include_reading_in_output: bool
) -> Unidirectional:

    match controller:
        case 'rnn':
            def controller(input_size):
                return SimpleRNN(
                    input_size=input_size,
                    hidden_units=hidden_units,
                    layers=layers,
                    dropout=dropout,
                    learned_initial_state=learned_initial_state
                )
        case 'lstm':
            def controller(input_size):
                return LSTM(
                    input_size=input_size,
                    hidden_units=hidden_units,
                    layers=layers,
                    dropout=dropout,
                    learned_initial_state=learned_initial_state
                )
        case _:
            raise ValueError
    controller_output_size = hidden_units
    reading_layer_sizes = None

    stack_name, stack_args = stack
    match stack_name:
        case 'stratification':
            stack_embedding_size, = stack_args
            return StratificationStackRNN(
                input_size=None,
                stack_embedding_size=stack_embedding_size,
                controller=controller,
                controller_output_size=controller_output_size,
                include_reading_in_output=include_reading_in_output,
                reading_layer_sizes=reading_layer_sizes
            )
        case 'superposition':
            stack_embedding_size, = stack_args
            return SuperpositionStackRNN(
                input_size=None,
                stack_embedding_size=stack_embedding_size,
                push_hidden_state=False,
                controller=controller,
                controller_output_size=controller_output_size,
                include_reading_in_output=include_reading_in_output,
                reading_layer_sizes=reading_layer_sizes
            )
        case 'nondeterministic':
            num_states, stack_alphabet_size = stack_args
            return NondeterministicStackRNN(
                input_size=None,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                controller=controller,
                controller_output_size=controller_output_size,
                include_reading_in_output=include_reading_in_output,
                reading_layer_sizes=reading_layer_sizes
            )
        case 'vector-nondeterministic':
            num_states, stack_alphabet_size, stack_embedding_size = stack_args
            return VectorNondeterministicStackRNN(
                input_size=None,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                stack_embedding_size=stack_embedding_size,
                controller=controller,
                controller_output_size=controller_output_size,
                include_reading_in_output=include_reading_in_output,
                reading_layer_sizes=reading_layer_sizes
            )
        case _:
            raise ValueError
