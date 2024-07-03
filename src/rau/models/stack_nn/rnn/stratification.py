import typing

import torch

from rau.tools.torch.layer import Layer
from rau.models.stack_nn.differentiable_stacks.stratification import StratificationStack

from .stack_rnn import StackRNN, StackRNNController, ReadingLayerSizes

class StratificationStackRNN(StackRNN):

    def __init__(self,
        input_size: int,
        stack_embedding_size: int,
        controller: StackRNNController,
        controller_output_size: int,
        include_reading_in_output: bool,
        reading_layer_sizes: ReadingLayerSizes = None
    ):
        super().__init__(
            input_size=input_size,
            stack_reading_size=stack_embedding_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
        self.stack_embedding_size = stack_embedding_size
        self.action_layer = Layer(
            controller_output_size,
            2,
            torch.nn.Sigmoid()
        )
        self.push_value_layer = Layer(
            controller_output_size,
            stack_embedding_size,
            torch.nn.Tanh()
        )

    def initial_stack(self, batch_size, sequence_length=None):
        t = next(self.parameters())
        return StratificationStack.new_empty(
            batch_size=batch_size,
            stack_embedding_size=self.stack_embedding_size,
            dtype=t.dtype,
            device=t.device
        )

    class State(StackRNN.State):

        def compute_stack(self, hidden_state, stack):
            actions = self.rnn.action_layer(hidden_state)
            push_value = self.rnn.push_value_layer(hidden_state)
            return stack.next(actions, push_value), actions
