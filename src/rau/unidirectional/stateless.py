import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .unidirectional import Unidirectional, ForwardResult
from .util import ensure_is_forward_result, unwrap_output_tensor

class StatelessUnidirectional(Unidirectional):
    r"""A sequential module that has no temporal recurrence, but applies some
    function to every timestep."""

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""Transform an input tensor for a single timestep.

        :param input_tensor: A tensor of size :math:`B \times \cdots`
            representing a tensor for a single timestep.
        :return: A tensor of size :math:`B \times cdots`.
        """
        raise NotImplementedError

    def forward_sequence(self,
        input_sequence: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""Transform a sequence of tensors.

        :param input_sequence: A tensor of size :math:`B \times n \times \cdots`
            representing a sequence of tensors.
        :return: A tensor of size :math:`B \times n \cdots`.
        """
        raise NotImplementedError

    def initial_output(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""Get the output of the initial state. By default, this simply
        raises an error.

        :param batch_size: Batch size.
        :return: A tensor of size :math:`B \times \cdots`.
        """
        raise ValueError(
            'tried to get the output of the initial state of a '
            'StatelessUnidirectional, but the output is not defined'
        )

    def transform_args(self,
        args: list[Any],
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> list[Any]:
        return args

    def transform_kwargs(self,
        kwargs: dict[str, Any],
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> dict[str, Any]:
        return kwargs

    @dataclasses.dataclass
    class State(Unidirectional.State):

        parent: 'StatelessUnidirectional'
        args: list[Any]
        kwargs: dict[str, Any]
        _batch_size: int | None
        input_tensor: torch.Tensor | None

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                _batch_size=None,
                input_tensor=input_tensor
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
            if self.input_tensor is None:
                return self.parent.initial_output(self._batch_size, *self.args, **self.kwargs)
            else:
                return self.parent.forward_single(self.input_tensor, *self.args, **self.kwargs)

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            if include_first:
                raise NotImplementedError(
                    'include_first=True in StatelessUnidirectional.State.forward() '
                    'is not implemented yet'
                )
            if return_output:
                output = self.parent.forward_sequence(input_sequence, *self.args, **self.kwargs)
            else:
                output = None
            if return_state:
                if input_sequence.size(1) == 0:
                    state = self
                else:
                    state = dataclasses.replace(
                        self,
                        _batch_size=None,
                        input_tensor=input_sequence[:, -1]
                    )
            else:
                state = None
            return unwrap_output_tensor(ForwardResult(
                output=output,
                extra_outputs=[],
                state=state
            ))

        def batch_size(self) -> int:
            if self.input_tensor is None:
                return self._batch_size
            else:
                return self.input_tensor.size(0)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            if self.input_tensor is None:
                # TODO Simply returning self would not change the batch size.
                # It's possible to work around this by running func() on a
                # dummy tensor.
                raise ValueError(
                    'cannot call transform_tensors() on initial state of '
                    'StatelessUnidirectional'
                )
            else:
                return dataclasses.replace(
                    self,
                    input_tensor=func(self.input_tensor)
                )

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        return self.State(
            parent=self,
            args=args,
            kwargs=kwargs,
            _batch_size=batch_size,
            input_tensor=None
        )

    @dataclasses.dataclass
    class ComposedState(Unidirectional.State):

        parent: 'StatelessUnidirectional'
        args: list[Any]
        kwargs: dict[str, Any]
        input_is_main: bool
        input_state: Unidirectional.State

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            # Just advance the first state, but don't bother computing its
            # output. This means outputs are computed lazily.
            return dataclasses.replace(
                self,
                input_state=self.input_state.next(input_tensor)
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, *tuple[Any, ...]]:
            return self.parent.forward_single(
                self.input_state.output(),
                *self.args,
                **self.kwargs
            )

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            # Get the outputs from the input module.
            first_result = ensure_is_forward_result(self.input_state.forward(
                input_sequence,
                include_first=self.input_is_main and include_first,
                return_state=return_state,
                return_output=return_output
            ))
            if return_output:
                # Compute the outputs in parallel.
                output = self.parent.forward_sequence(
                    first_result.output,
                    *self.args,
                    **self.kwargs
                )
            else:
                output = None
            if return_state:
                new_state = dataclasses.replace(
                    self,
                    input_state=first_result.state
                )
            else:
                new_state = None
            return unwrap_output_tensor(ForwardResult(
                output=output,
                extra_outputs=first_result.extra_outputs,
                state=new_state
            ))

        def batch_size(self) -> int:
            return self.input_state.batch_size()

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                input_state=self.input_state.transform_tensors(func)
            )

    def initial_composed_state(self,
        input_module: Unidirectional,
        input_state: Unidirectional.State,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        if self._composable_is_main:
            raise NotImplementedError(
                'composing with a StatelessUnidirectional marked as main and '
                'using iterative mode is not implemented yet'
            )
        return self.ComposedState(
            parent=self,
            args=args,
            kwargs=kwargs,
            input_is_main=input_module._composable_is_main,
            input_state=input_state
        )

class StatelessLayerUnidirectional(StatelessUnidirectional):

    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.func(input_tensor, *args, **kwargs)

    def forward_sequence(self,
        input_sequence: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.func(input_sequence, *args, **kwargs)

class StatelessReshapingLayerUnidirectional(StatelessLayerUnidirectional):

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.func(input_tensor.unsqueeze(1), *args, **kwargs).squeeze(1)
