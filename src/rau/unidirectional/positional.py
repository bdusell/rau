import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .unidirectional import Unidirectional, ForwardResult
from .util import unwrap_output_tensor

class PositionalUnidirectional(Unidirectional):

    def forward_from_position(self,
        input_sequence: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        r"""Compute the outputs for a sequence of inputs, starting at a certain
        position.

        :param input_sequence: A tensor of size :math:`B \times n \times \cdots`
            representing a sequence of input tensors.
        :param position: An index indicating the timestep corresponding to the
            first input of ``input_sequence``. The first timestep has index 0.
        :return: A tensor of size :math:`B \times n' \times \cdots`
            representing a sequence of output tensors.
        """
        raise NotImplementedError

    def forward_at_position(self,
        input_tensor: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        r"""Compute the output for a single input at a certain position.

        :param input_tensor: A tensor of size :math:`B \times \cdots`
            representing an input tensor for a single timestep.
        :param position: An index indicating the current timestep. The first
            timestep has index 0.
        :return: A tensor of size :math:`B \times \cdots` representing the
            output tensor corresponding to the input tensor.
        """
        raise NotImplementedError

    @dataclasses.dataclass
    class State(Unidirectional.State):

        parent: 'PositionalUnidirectional'
        position: int
        _batch_size: int | None
        input_tensor: torch.Tensor | None

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                position=self.position + 1,
                _batch_size=None,
                input_tensor=input_tensor
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, *tuple[Any, ...]]:
            if self.input_tensor is None:
                raise ValueError(
                    'initial state of PositionalUnidirectional does not have '
                    'an output'
                )
            return self.parent.forward_at_position(self.input_tensor, self.position - 1)

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            if return_output:
                new_position = self.position
                new_input_sequence = input_sequence
                if include_first:
                    if self.input_tensor is None:
                        raise ValueError(
                            'initial state of PositionalUnidirectional does not have '
                            'an output'
                        )
                    new_position -= 1
                    new_input_sequence = torch.concat([
                        self.input_tensor[:, None], input_sequence
                    ], dim=1)
                output = self.parent.forward_from_position(
                    new_input_sequence,
                    new_position
                )
            else:
                output = None
            if return_state:
                if input_sequence.size(1) == 0:
                    return self
                else:
                    state = dataclasses.replace(
                        self,
                        position=self.position + input_sequence.size(1),
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
                    'PositionalUnidirectional'
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
            position=0,
            _batch_size=batch_size,
            input_tensor=None
        )

    # TODO Implement lazy outputs.
