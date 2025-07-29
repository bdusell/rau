import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from rau.unidirectional import Unidirectional, ForwardResult
from rau.tools.torch.layer import FeedForward
from rau.models.stack_nn.differentiable_stacks.stack import DifferentiableStack

def cached_tensor(method):
    name = method.__name__
    def wrapped_method(self):
        result = self._cached_tensors.get(name)
        if result is None:
            result = self._cached_tensors[name] = method(self)
        return result
    return wrapped_method

StackRNNController = Callable[[int], torch.nn.Module]
ReadingLayerSizes = list[int | None] | None

class StackRNN(Unidirectional):

    def __init__(self,
        input_size: int,
        stack_reading_size: int,
        controller: StackRNNController,
        controller_output_size: int,
        include_reading_in_output: bool,
        reading_layer_sizes: ReadingLayerSizes = None
    ) -> None:
        """
        :param input_size: The size of the input vectors to this module.
        :param stack_reading_size: The size of the reading vector returned
            from the stack module.
        :param controller: A constructor function that takes an input size and
            returns a Unidirectional implementing the controller.
        :param controller_output_size: The size of the output vectors from the
            controller.
        :param include_reading_in_output: Whether to include the stack reading
            (after applying any layers) in the output along with the output of
            the controller (concatenated into one vector).
        :param reading_layer_sizes: An optional list specifying the sizes of
            hidden layers to apply between the stack reading and the
            controller. A size of ``None`` indicates that the
            ``stack_reading_size`` should be used.
        """
        super().__init__()
        if reading_layer_sizes:
            reading_layer_sizes = [x if x is not None else stack_reading_size for x in reading_layer_sizes]
            self.reading_layer = FeedForward(
                input_size=stack_reading_size,
                layer_sizes=reading_layer_sizes,
                activation=torch.nn.Tanh()
            )
            reading_layer_output_size = self.reading_layer.output_size()
        else:
            self.reading_layer = torch.nn.Identity()
            reading_layer_output_size = stack_reading_size
        self.include_reading_in_output = include_reading_in_output
        self.controller = controller(input_size + reading_layer_output_size)
        self._output_size = controller_output_size
        if self.include_reading_in_output:
            self._output_size += reading_layer_output_size

    def output_size(self) -> int:
        return self._output_size

    def forward(self,
        input_sequence: torch.Tensor,
        *args: Any,
        initial_state: 'Unidirectional.State | None' = None,
        return_state: bool = False,
        include_first: bool = True,
        **kwargs: Any
    ) -> torch.Tensor | ForwardResult:
        # Automatically add the length of the input as an extra argument.
        return super().forward(
            input_sequence,
            *args,
            initial_state=initial_state,
            return_state=return_state,
            include_first=include_first,
            sequence_length=input_sequence.size(1),
            **kwargs
        )

    @dataclasses.dataclass
    class State(Unidirectional.State):

        rnn: 'StackRNN'
        num_inputs_read: int
        sequence_length: int | None
        hidden_state: Unidirectional.State
        previous_stack: DifferentiableStack | None
        return_actions: bool
        return_readings: bool
        stack_args: list[Any]
        stack_kwargs: dict[str, Any]

        def __post_init__(self):
            self._cached_tensors = {}

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            # Get the current state of the differentiable stack.
            stack = self.get_stack()
            # Get the stack reading, possibly passing it through a feedforward
            # network first.
            reading_layer_output = self.get_reading_layer_output()
            # Concatenate the input and stack reading vectors and pass those as
            # the next input to the controller.
            controller_input = torch.cat((input_tensor, reading_layer_output), dim=1)
            next_hidden_state = self.hidden_state.next(controller_input)
            return dataclasses.replace(
                self,
                num_inputs_read=self.num_inputs_read + 1,
                hidden_state=next_hidden_state,
                previous_stack=stack,
                stack_args=None,
                stack_kwargs=None
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, *tuple[Any, ...]]:
            output = self.get_output()
            extras = []
            if self.return_actions:
                if (
                    self.sequence_length is None or
                    self.num_inputs_read < self.sequence_length
                ):
                    actions = self.get_actions()
                else:
                    actions = None
                extras.append(actions)
            if self.return_readings:
                if (
                    self.sequence_length is None or
                    self.num_inputs_read < self.sequence_length
                ):
                    reading = self.get_reading()
                else:
                    reading = None
                extras.append(reading)
            if extras:
                return (output, *extras)
            else:
                return output

        def get_actions(self) -> Any:
            stack, actions = self.get_stack_and_actions()
            return actions

        def get_stack(self) -> DifferentiableStack:
            stack, actions = self.get_stack_and_actions()
            return stack

        @cached_tensor
        def get_stack_and_actions(self) -> tuple[DifferentiableStack, Any]:
            if self.previous_stack is None:
                stack = self.rnn.initial_stack(
                    self.hidden_state.batch_size(),
                    self.sequence_length,
                    *self.stack_args,
                    **self.stack_kwargs
                )
                actions = None
            else:
                stack, actions = self.compute_stack(
                    self.get_hidden_state_output(),
                    self.previous_stack
                )
                # The previous stack is no longer needed now.
                self.previous_stack = None
            return stack, actions

        @cached_tensor
        def get_reading(self) -> torch.Tensor:
            return self.get_stack().reading()

        @cached_tensor
        def get_reading_layer_output(self) -> torch.Tensor:
            return self.rnn.reading_layer(self.get_reading())

        @cached_tensor
        def get_hidden_state_output(self) -> torch.Tensor:
            return self.hidden_state.output()

        @cached_tensor
        def get_output(self) -> torch.Tensor:
            if self.rnn.include_reading_in_output:
                return torch.concat([
                    self.get_hidden_state_output(),
                    self.get_reading_layer_output()
                ], dim=1)
            else:
                return self.get_hidden_state_output()

        def batch_size(self) -> int:
            return self.hidden_state.batch_size()

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            result = dataclasses.replace(
                self,
                hidden_state=self.hidden_state.transform_tensors(func),
                previous_stack=self.previous_stack.transform_tensors(func) if self.previous_stack is not None else None
            )
            for k, v in self._cached_tensors.items():
                result._cached_tensors[k] = func(v)
            return result

        def compute_stack(self,
            hidden_state: Unidirectional.State,
            stack: DifferentiableStack
        ) -> DifferentiableStack:
            raise NotImplementedError

    def initial_state(self,
        batch_size: int,
        *args: Any,
        sequence_length: int | None = None,
        return_actions: bool = False,
        return_readings: bool = False,
        **kwargs: Any
    ) -> Unidirectional.State:
        """Get the initial state of the stack RNN.

        :param sequence_length: Used to determine when the last timestep is
            reached, which may avoid some unnecessary computation. The actions
            and reading will not be returned for the last timestep if they are
            not needed.
        :param return_actions: If true, then the output at each timestep will
            also include the stack actions that were emitted at that timestep.
            Note that the stack actions for timestep 0 are always ``None``.
        :param return_readings: If true, then the output at each timestep will
            also include the stack reading that was emitted just before the
            current timestep.
        :param args: Will be passed to :py:meth:`initial_stack`.
        :param kwargs: Will be passed to :py:meth:`initial_stack`.
        """
        return self.State(
            rnn=self,
            num_inputs_read=0,
            sequence_length=sequence_length,
            hidden_state=self.controller.initial_state(batch_size),
            # There is no "previous stack" for the initial hidden state, so
            # set it to None. It will call initial_stack() to supply the stack
            # for the next timestep.
            previous_stack=None,
            return_actions=return_actions,
            return_readings=return_readings,
            stack_args=args,
            stack_kwargs=kwargs
        )

    def initial_stack(self,
        batch_size: int,
        sequence_length: int | None,
        *args: Any,
        **kwargs: Any
    ) -> DifferentiableStack:
        raise NotImplementedError
