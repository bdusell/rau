import math
from collections.abc import Sequence
from typing import Literal

import torch

from rau.tools.torch.layer import Layer, MultiLayer
from rau.models.stack_nn.differentiable_stacks.superposition import SuperpositionStack

from .stack_rnn import StackRNN, StackRNNController, ReadingLayerSizes

class SuperpositionStackRNN(StackRNN):
    """The superposition stack RNN proposed by Joulin and Mikolov (2015). It
    consists of an RNN controller connected to a differentiable superposition
    stack data structure."""

    def __init__(self,
        input_size: int,
        stack_embedding_size: int | Sequence[int],
        push_hidden_state: bool,
        controller: StackRNNController,
        controller_output_size: int,
        include_reading_in_output: bool,
        max_stack_depth: int | Literal[math.inf]=math.inf,
        reading_layer_sizes: ReadingLayerSizes=None
    ):
        """Construct a new superposition stack RNN.

        :param input_size: The size of the vectors provided as input to this
            RNN.
        :param stack_embedding_size: If a single integer is given, this
            determines the size of the vector elements in the stack. All of the
            stack actions will be synchronized across all of the units of these
            vectors. If a sequence of integers if given, then multiple stacks
            will be simulated, where the number of integers determines the
            number of stacks, and each integer determines the size of the
            vector elements of each stack. The stack actions will be
            synchronized across all units within each stack, but across
            different stacks.
        :param push_hidden_state: Whether to push the hidden state of the
            controller or to learn a projection for pushed vectors
            automatically.
        :param controller: Constructor for the RNN controller.
        """
        if isinstance(stack_embedding_size, int):
            stack_embedding_sizes = (stack_embedding_size,)
        else:
            stack_embedding_sizes = tuple(stack_embedding_size)
        total_stack_embedding_size = sum(stack_embedding_sizes)
        super().__init__(
            input_size=input_size,
            stack_reading_size=total_stack_embedding_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
        self.stack_embedding_sizes = stack_embedding_sizes
        self.total_stack_embedding_size = total_stack_embedding_size
        self.action_layer = MultiLayer(
            input_size=controller_output_size,
            output_size=3,
            num_layers=len(stack_embedding_sizes),
            activation=torch.nn.Softmax(dim=2)
        )
        if push_hidden_state:
            hidden_state_size = controller_output_size
            if total_stack_embedding_size != hidden_state_size:
                raise ValueError(
                    f'push_hidden_state is True, but the total stack '
                    f'embedding size ({total_stack_embedding_size}) does not '
                    f'match the output size of the controller '
                    f'({hidden_state_size})')
            self.push_value_layer = torch.nn.Identity()
        else:
            self.push_value_layer = Layer(
                controller_output_size,
                total_stack_embedding_size,
                torch.nn.Sigmoid()
            )
        self.max_stack_depth = max_stack_depth

    def forward(self, input_sequence, *args, return_state=False, **kwargs):
        # Automatically use the sequence length to optimize the stack
        # computation. Don't use it if returning the stack state.
        max_sequence_length = math.inf if return_state else input_sequence.size(1)
        return super().forward(
            input_sequence,
            *args,
            return_state=return_state,
            max_sequence_length=max_sequence_length,
            **kwargs)

    class State(StackRNN.State):

        def compute_stack(self, hidden_state, stack):
            # unexpanded_actions : batch_size x num_stacks x num_actions
            unexpanded_actions = self.rnn.action_layer(hidden_state)
            # actions : batch_size x total_stack_embedding_size x num_actions
            actions = expand_actions(unexpanded_actions, self.rnn.stack_embedding_sizes)
            # push_prob, etc. : batch_size x total_stack_embedding_size
            push_prob, pop_prob, noop_prob = torch.unbind(actions, dim=2)
            # push_value : batch_size x total_stack_embedding_size
            push_value = self.rnn.push_value_layer(hidden_state)
            return stack.next(push_prob, pop_prob, noop_prob, push_value), unexpanded_actions

    def initial_stack(self,
        batch_size,
        sequence_length=None,
        max_sequence_length: int | Literal[math.inf]=math.inf,
        stack_constructor=SuperpositionStack.new_empty
    ):
        """
        If the sequence length is known, passing it via `max_sequence_length`
        can be used to reduce the time and space required by the stack by half.
        """
        t = next(self.parameters())
        return stack_constructor(
            batch_size=batch_size,
            stack_embedding_size=self.total_stack_embedding_size,
            max_sequence_length=max_sequence_length,
            max_depth=self.max_stack_depth,
            dtype=t.dtype,
            device=t.device
        )

def expand_actions(actions, sizes):
    # actions : batch_size x num_stacks x num_actions
    # sizes : num_stacks x [int]
    # return : batch_size x sum(sizes) x num_actions
    batch_size, num_stacks, num_actions = actions.size()
    if len(sizes) == 1:
        return actions.expand(batch_size, sizes[0], num_actions)
    else:
        return torch.cat([
            actions_i[:, None, :].expand(batch_size, size_i, num_actions)
            for actions_i, size_i in zip(torch.unbind(actions, dim=1), sizes)
        ], dim=1)
