from typing import Literal

import torch

from rau.models.stack_nn.differentiable_stacks.semiring import log
from rau.models.stack_nn.differentiable_stacks.vector_nondeterministic import (
    VectorNondeterministicStack
)

from .stack_rnn import StackRNN, StackRNNController, ReadingLayerSizes
from .nondeterministic import NondeterministicStackRNN

class VectorNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self,
        input_size: int | None,
        num_states: int,
        stack_alphabet_size: int,
        stack_embedding_size: int,
        controller: StackRNNController,
        controller_output_size: int,
        include_reading_in_output: bool,
        normalize_transition_weights: bool = False,
        reading_layer_sizes: ReadingLayerSizes = None,
        bottom_vector: Literal['learned', 'one', 'zero'] = 'learned'
    ) -> None:
        Q = num_states
        S = stack_alphabet_size
        m = stack_embedding_size
        super().__init__(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            normalize_transition_weights=normalize_transition_weights,
            normalize_reading=True,
            reading_layer_sizes=reading_layer_sizes,
            stack_reading_size=Q * S * m
        )
        self.stack_embedding_size = stack_embedding_size
        self.pushed_vector_layer = torch.nn.Sequential(
            torch.nn.Linear(
                controller_output_size,
                stack_embedding_size
            ),
            torch.nn.LogSigmoid()
        )
        # This parameter is the learned embedding that always sits at the
        # bottom of the stack. It is the input to a sigmoid operation, so the
        # vector used in the stack will be in (0, 1).
        self.bottom_vector_type = bottom_vector
        if bottom_vector == 'learned':
            self.bottom_vector = torch.nn.Parameter(torch.zeros((m,)))
        elif bottom_vector in ('one', 'zero'):
            pass
        else:
            raise ValueError(f'unknown bottom vector option: {bottom_vector!r}')

    def pushed_vector(self, hidden_state):
        return self.pushed_vector_layer(hidden_state)

    def get_bottom_vector(self, semiring):
        if self.bottom_vector_type == 'learned':
            if semiring is not log:
                raise NotImplementedError
            return torch.nn.functional.logsigmoid(self.bottom_vector)
        elif self.bottom_vector_type == 'one':
            tensor = next(self.parameters())
            return semiring.ones((self.stack_embedding_size,), like=tensor)
        elif self.bottom_vector_type == 'zero':
            return None
        else:
            raise ValueError

    def get_new_stack(self, batch_size, sequence_length, semiring, block_size):
        t = next(self.parameters())
        # If the stack reading is not included in the output, then the last
        # timestep is not needed.
        if not self.include_reading_in_output and sequence_length is not None:
            sequence_length -= 1
        return VectorNondeterministicStack.new_empty(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            stack_embedding_size=self.stack_embedding_size,
            sequence_length=sequence_length,
            bottom_vector=self.get_bottom_vector(semiring),
            block_size=block_size,
            dtype=t.dtype,
            device=t.device,
            semiring=semiring
        )

    class State(StackRNN.State):

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = self.rnn.operation_log_scores(hidden_state)
            pushed_vector = self.rnn.pushed_vector(hidden_state)
            actions = (push, repl, pop, pushed_vector)
            stack.update(push, repl, pop, pushed_vector)
            return stack, actions

        def transform_stack_actions(self, actions, func):
            return tuple(map(func, actions))
