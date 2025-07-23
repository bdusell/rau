from collections.abc import Callable, Iterable
import dataclasses
from typing import Any, overload

import torch

from rau.tools.torch.compose import Composable, Composed

@dataclasses.dataclass
class ForwardResult:
    r"""The output of a call to :py:meth:`Unidirectional.forward` or
    :py:meth:`Unidirectional.State.forward`."""

    output: torch.Tensor | None
    r"""The main output tensor of the module."""
    extra_outputs: list[list[Any]]
    r"""A list of extra outputs returned alongside the main output."""
    state: 'Unidirectional.State | None'
    r"""An optional state representing the updated state of the module after
    reading the inputs."""

class Unidirectional(torch.nn.Module):
    """An API for unidirectional sequential neural networks (including RNNs
    and transformer decoders).

    Let :math:`B` be batch size, and :math:`n` be the length of the input
    sequence.
    """

    def __init__(self, main: bool = False, tags: Iterable[str] | None = None):
        super().__init__()
        self._composable_is_main = main
        self._composable_tags = dict.fromkeys(tags) if tags is not None else {}

    def forward(self,
        input_sequence: torch.Tensor,
        *args: Any,
        initial_state: 'Unidirectional.State | None' = None,
        return_state: bool = False,
        include_first: bool = True,
        **kwargs: Any
    ) -> torch.Tensor | ForwardResult:
        r"""Run this module on an entire sequence of inputs all at once.

        This can often be done more efficiently than processing each input one
        by one.

        :param input_sequence: A :math:`B \times n \times \cdots` tensor
            representing a sequence of :math:`n` input tensors.
        :param initial_state: An optional initial state to use instead of the
            default initial state created by :py:meth:`initial_state`.
        :param return_state: Whether to return the last :py:class:`State` of
            the module as an additional output. This state can be used to
            initialize a subsequent run.
        :param include_first: Whether to prepend an extra tensor to the
            beginning of the output corresponding to a prediction for the
            first element in the input. If ``include_first`` is true, then the
            length of the output tensor will be :math:`n + 1`. Otherwise, it
            will be :math:`n`.
        :param args: Extra arguments passed to :py:meth:`initial_state`.
        :param kwargs: Extra arguments passed to :py:meth:`initial_state`.
        :return: A :py:class:`~torch.Tensor` or a :py:class:`ForwardResult` that
            contains the output tensor. The output tensor will be of size
            :math:`B \times n+1 \times \cdots` if ``include_first`` is true and
            :math:`B \times n \times \cdots` otherwise. If
            :py:meth:`Unidirectional.State.output` returns extra outputs at
            each timestep, then they will be aggregated over all timesteps and
            returned as :py:class:`list`\ s in :py:attr:`ForwardResult.extra_outputs`.
            If ``return_state`` is true, then the final :py:class:`State` will
            be returned in :py:attr:`ForwardResult.state`. If there are no extra
            outputs and there is no state to return, just the output tensor is
            returned.
        """
        # input_sequence: B x n x ...
        if initial_state is not None:
            if not isinstance(initial_state, self.State):
                raise TypeError(f'initial_state must be of type {self.State.__name__}')
            state = initial_state
        else:
            batch_size = input_sequence.size(0)
            state = self.initial_state(batch_size, *args, **kwargs)
        # return : B x n x ...
        return state.forward(
            input_sequence,
            return_state=return_state,
            include_first=include_first
        )

    @overload
    def __or__(self, other: torch.nn.Module) -> Composed:
        ...
    @overload
    def __or__(self, other: 'Unidirectional') -> 'Unidirectional':
        ...
    def __or__(self, other):
        r"""The ``|`` operator is overridden to compose two Unidirectionals."""
        if isinstance(other, Unidirectional):
            from .composed import ComposedUnidirectional
            return ComposedUnidirectional(self, other)
        else:
            return self.as_composable() | other

    def as_composable(self) -> Composable:
        return Composable(
            self,
            main=self._composable_is_main,
            tags=self._composable_tags.keys()
        )

    class State:
        r"""Represents the hidden state of the module after processing a certain
        number of inputs."""

        def next(self, input_tensor: torch.Tensor) -> 'Unidirectional.State':
            r"""Feed an input to this hidden state and produce the next hidden
            state.

            :param input_tensor: A tensor of size :math:`B \times \cdots`,
                representing an input for a single timestep.
            """
            raise NotImplementedError

        def output(self) -> torch.Tensor | tuple[torch.Tensor, *tuple[Any, ...]]:
            r"""Get the output associated with this state.

            For example, this can be the hidden state vector itself, or the
            hidden state passed through an affine transformation.

            The return value is either a tensor or a tuple whose first element
            is a tensor. The other elements of the tuple can be used to return
            extra outputs.

            :return: A :math:`B \times \cdots` tensor, or a tuple whose first
                element is a tensor. The other elements of the tuple can
                contain extra outputs. If there are any extra outputs, then
                the output of :py:meth:`forward` and
                :py:meth:`Unidirectional.forward` will contain the same
                number of extra outputs, where each extra output is a
                :py:class:`list` containing all the outputs across all
                timesteps.
            """
            raise NotImplementedError

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            r"""Like :py:meth:`Unidirectional.forward`, but start with this
            state as the initial state.

            This can often be done more efficiently than using :py:meth:`next`
            iteratively.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor,
                representing :math:`n` input tensors.
            :param return_state: Whether to return the last :py:class:`State`
                of the module.
            :param include_first: Whether to prepend an extra tensor to the
                beginning of the output corresponding to an output from this
                state, before reading the first input.
            :return: See :py:meth:`Unidirectional.forward`.
            """
            from .util import unwrap_output_tensor
            if return_output:
                outputs = []
            state = self
            if return_output and include_first:
                outputs.append(state.output())
            for input_tensor in input_sequence.transpose(0, 1):
                state = state.next(input_tensor)
                if return_output:
                    outputs.append(state.output())
            if return_output:
                output, extra_outputs = _stack_outputs(outputs)
            else:
                output = None
                extra_outputs = []
            return unwrap_output_tensor(ForwardResult(
                output,
                extra_outputs,
                state if return_state else None
            ))

        def fastforward(self, input_sequence: torch.Tensor) -> 'Unidirectional.State':
            r"""Feed a sequence of inputs to this state and return the
            resulting state.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor,
                representing :math:`n` input tensors.
            :return: Updated state after reading ``input_sequence``.
            """
            return self.forward(
                input_sequence,
                include_first=False,
                return_state=True,
                return_output=False
            ).state

        def batch_size(self) -> int:
            r"""Get the batch size of the tensors in this state."""
            raise NotImplementedError

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> 'Unidirectional.State':
            r"""Return a copy of this state with all tensors passed through a
            function.

            :param func: A function that will be applied to all tensors in this
                state.
            """
            raise NotImplementedError

        def detach(self) -> 'Unidirectional.State':
            r"""Return a copy of this state with all tensors detached."""
            return self.transform_tensors(lambda x: x.detach())

        def slice_batch(self, s: slice) -> 'Unidirectional.State':
            r"""Return a copy of this state with only certain batch elements
            included, determined by the slice ``s``.

            :param s: The slice object used to determine which batch elements
                to keep.
            """
            return self.transform_tensors(lambda x: x[s, ...])

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> 'Unidirectional.State':
        r"""Get the initial state of the model.

        :param batch_size: Batch size.
        :param args: Extra arguments passed from :py:meth:`forward`.
        :param kwargs: Extra arguments passed from :py:meth:`forward`.
        :return: A state.
        """
        raise NotImplementedError

    @dataclasses.dataclass
    class StatefulComposedState(State):

        parent: 'Unidirectional'
        first_is_main: bool
        first_state: 'Unidirectional.State'
        second_state: 'Unidirectional.State'

        def next(self, input_tensor: torch.Tensor) -> 'Unidirectional.State':
            # Always advance the first state and evaluate its output, then feed
            # that as input to the second state. No lazy evaluation here.
            new_first_state = self.first_state.next(input_tensor)
            new_second_state = self.second_state.next(new_first_state.output())
            return dataclasses.replace(
                self,
                first_state=new_first_state,
                second_state=new_second_state
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, *tuple[Any, ...]]:
            return self.second_state.output()

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            from .util import unwrap_output_tensor, ensure_is_forward_result
            first_result = ensure_is_forward_result(self.first_state.forward(
                input_sequence,
                include_first=self.first_is_main and include_first,
                return_state=return_state,
                return_output=True
            ))
            second_result = ensure_is_forward_result(self.second_state.forward(
                first_result.output,
                include_first=self.parent._composable_is_main and include_first,
                return_state=return_state,
                return_output=return_output
            ))
            if return_state:
                new_state = dataclasses.replace(
                    self,
                    first_state=first_result.state,
                    second_state=second_result.state
                )
            else:
                new_state = None
            return ForwardResult(
                output=second_result.output,
                extra_outputs=first_result.extra_outputs + second_result.extra_outputs,
                state=new_state
            )

        def batch_size(self) -> int:
            return self.second_state.batch_size()

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> 'Unidirectional.State':
            return Unidirectional.StatefulComposedState(
                self.first_state.transform_tensors(func),
                self.second_state.transform_tensors(func)
            )

    def initial_composed_state(self,
        input_module: 'Unidirectional',
        input_state: 'Unidirectional.State',
        *args: Any,
        **kwargs: Any
    ) -> 'Unidirectional.ComposedState':
        first_state = input_state
        second_state = self.initial_state(input_state.batch_size(), *args, **kwargs)
        # If the input module is main, then feed its initial output to this
        # module as the first input.
        if input_module._composable_is_main:
            second_state = second_state.next(first_state.output())
        return Unidirectional.StatefulComposedState(
            parent=self,
            first_is_main=input_module._composable_is_main,
            first_state=first_state,
            second_state=second_state
        )

    def main(self) -> 'Unidirectional':
        r"""Mark this module as main.

        :return: Self.
        """
        self._composable_is_main = True
        return self

    def tag(self, tag: str) -> 'Unidirectional':
        r"""Add a tag to this module for argument routing.

        :param tag: Tag name.
        :return: Self.
        """
        self._composable_tags[tag] = None
        return self

def _stack_outputs(
    outputs: Iterable[torch.Tensor | tuple[torch.Tensor, ...]]
) -> ForwardResult:
    it = iter(outputs)
    first = next(it)
    if isinstance(first, tuple):
        output, *extra = first
        output_list = [output]
        extra_lists = [[e] for e in extra]
        for output_t in it:
            if not isinstance(output_t, tuple):
                raise TypeError
            output, *extra = output_t
            if not isinstance(output, torch.Tensor):
                raise TypeError
            output_list.append(output)
            for extra_list, extra_item in zip(extra_lists, extra, strict=True):
                extra_list.append(extra_item)
    elif isinstance(first, torch.Tensor):
        output_list = [first]
        for output_t in it:
            if not isinstance(output_t, torch.Tensor):
                raise TypeError
            output_list.append(output_t)
        extra_lists = []
    else:
        raise TypeError
    output_tensor = torch.stack(output_list, dim=1)
    return output_tensor, extra_lists
