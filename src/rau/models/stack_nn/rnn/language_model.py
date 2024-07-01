from rau.unidirectional import Unidirectional
from rau.models.rnn.language_model import get_rnn_language_model_no_output_dropout

from .parse import StackRNNController, StackRNNStack
#from .stratification import StratificationStackRNN
#from .superposition import SuperpositionStackRNN
#from .nondeterministic import NondeteterministicStackRNN
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
    use_padding: bool
) -> Unidirectional:
    return get_rnn_language_model_no_output_dropout(
        get_stack_rnn_recurrence(
            controller=controller,
            stack=stack,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            learned_hidden_state=learned_hidden_state
        ),
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        hidden_units=hidden_units,
        dropout=dropout,
        use_padding=use_padding
    )

def get_stack_rnn_recurrence(
    controller: StackRNNController,
    stack: StackRNNStack,
    hidden_units: int,
    layers: int,
    dropout: float,
    learned_hidden_state: bool
) -> Unidirectional:
    stack_name, stack_args = stack
    if stack_name == 'stratification':
        stack_embedding_size, = stack_args
        return StratificationStackRNN(
            controller=controller,
            hidden_units=hidden_units,
            layers=layers,
            learned_hidden_state=learned_hidden_state,
            dropout=dropout,
            stack_embedding_size=stack_embedding_size
        )
    elif stack_name == 'superposition':
        stack_embedding_size, = stack_args
        return SuperpositionStackRNN(
            controller=controller,
            hidden_units=hidden_units,
            layers=layers,
            learned_hidden_state=learned_hidden_state,
            dropout=dropout,
            stack_embedding_size=stack_embedding_size
        )
    elif stack_name == 'nondeterministic':
        num_states, stack_alphabet_size = stack_args
        return NondeterministicStackRNN(
            controller=controller,
            hidden_units=hidden_units,
            layers=layers,
            learned_hidden_state=learned_hidden_state,
            dropout=dropout,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size
        )
    elif stack_name == 'vector-nondeterministic':
        num_states, stack_alphabet_size, stack_embedding_size = stack_args
        return VectorNondeterministicStackRNN(
            controller=controller,
            hidden_units=hidden_units,
            layers=layers,
            learned_hidden_state=learned_hidden_state,
            dropout=dropout,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size
        )
    else:
        raise ValueError
