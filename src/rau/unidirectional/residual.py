import dataclasses
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .unidirectional import Unidirectional, ForwardResult
from .stateless import StatelessUnidirectional
from .util import unwrap_output_tensor, ensure_is_forward_result

class ResidualUnidirectional(Unidirectional):

    def __init__(self, module: Unidirectional) -> None:
        super().__init__()
        if isinstance(module, StatelessUnidirectional):
            warnings.warn(
                'you are wrapping a StatelessUnidirectional with '
                'ResidualUnidirectional; for efficiency, consider wrapping it '
                'in StatelessResidualUnidirectional instead'
            )
        self.wrapped_module = module

    @dataclasses.dataclass
    class State(Unidirectional.State):

        input_tensor: torch.Tensor | None
        wrapped_state: Unidirectional.State

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                input_tensor=input_tensor,
                wrapped_state=self.wrapped_state.next(input_tensor)
            )

        def output(self) -> torch.Tensor | tuple[torch.Tensor, *tuple[Any, ...]]:
            if self.input_tensor is None:
                raise ValueError(
                    'the initial state of a ResidualUnidirectional has no '
                    'input, so it has no output'
                )
            return self.input_tensor + self.wrapped_state.output()

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            if include_first:
                raise NotImplementedError
            wrapped_result = ensure_is_forward_result(self.wrapped_state.forward(
                input_sequence,
                include_first=False,
                return_state=return_state,
                return_output=return_output
            ))
            if return_output:
                output = input_sequence + wrapped_result.output
            else:
                output = None
            if return_state:
                if input_sequence.size(1) == 0:
                    state = self
                else:
                    state = dataclasses.replace(
                        self,
                        input_tensor=input_sequence[:, -1],
                        wrapped_state=wrapped_result.state
                    )
            else:
                state = None
            return unwrap_output_tensor(ForwardResult(
                output=output,
                extra_outputs=wrapped_result.extra_outputs,
                state=state
            ))

        def batch_size(self) -> int:
            return self.wrapped_state.batch_size()

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                input_tensor=func(self.input_tensor) if self.input_tensor is not None else None,
                wrapped_state=self.wrapped_state.transform_tensors(func)
            )

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        return self.State(
            input_tensor=None,
            wrapped_state=self.wrapped_module.initial_state(
                batch_size,
                *args,
                **kwargs
            )
        )

class StatelessResidualUnidirectional(StatelessUnidirectional):

    def __init__(self, module: StatelessUnidirectional) -> None:
        super().__init__()
        if not isinstance(module, StatelessUnidirectional):
            raise TypeError
        self.wrapped_module = module

    def forward_single(self,
        input_tensor: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return (
            input_tensor +
            self.wrapped_module.forward_single(input_tensor, *args, **kwargs)
        )

    def forward_sequence(self,
        input_sequence: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return (
            input_sequence +
            self.wrapped_module.forward_sequence(input_sequence, *args, **kwargs)
        )
