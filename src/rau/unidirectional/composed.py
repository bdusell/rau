from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import itertools
from typing import Any

import torch

from .unidirectional import Unidirectional, ForwardResult
from .util import unwrap_output_tensor, ensure_is_forward_result

class ComposedUnidirectional(Unidirectional):
    """Stacks one undirectional model on another, so that the outputs of the
    first are fed as inputs to the second."""

    def __init__(self, first: Unidirectional, second: Unidirectional):
        super().__init__(
            first._composable_is_main or second._composable_is_main,
            itertools.chain(
                first._composable_tags.keys(),
                second._composable_tags.keys()
            )
        )
        self.first = first
        self.second = second
        # Normalize the composition order so that it is always left-associative.
        # This is important for lazy evaluation to work properly.
        if isinstance(self.second, ComposedUnidirectional):
            # TODO Move tags from self.second to its children?
            for m in _list_modules(self.second.first):
                self.first = self.first | m
            self.second = self.second.second

    def forward(self,
        input_sequence: torch.Tensor,
        *args: Any,
        initial_state: Unidirectional.State | None = None,
        return_state: bool = False,
        include_first: bool = True,
        tag_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any
    ) -> torch.Tensor | ForwardResult:
        if (args or kwargs) and not self._composable_is_main:
            raise ValueError('this module does not accept extra args or kwargs')
        if initial_state is None and not return_state:
            first_args, first_kwargs = get_composed_args(self.first, args, kwargs, tag_kwargs, include_first)
            second_args, second_kwargs = get_composed_args(self.second, args, kwargs, tag_kwargs, include_first)
            first_result = ensure_is_forward_result(self.first(
                input_sequence,
                *first_args,
                return_state=False,
                **first_kwargs
            ))
            second_result = ensure_is_forward_result(self.second(
                first_result.output,
                *second_args,
                return_state=False,
                **second_kwargs
            ))
            return unwrap_output_tensor(ForwardResult(
                output=second_result.output,
                extra_outputs=first_result.extra_outputs + second_result.extra_outputs,
                state=None
            ))
        else:
            return super().forward(
                input_sequence,
                *args,
                initial_state=initial_state,
                return_state=return_state,
                tag_kwargs=tag_kwargs,
                **kwargs
            )

    def initial_state(self,
        batch_size: int,
        *args: Any,
        tag_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any
    ) -> 'Unidirectional.State':
        if (args or kwargs) and not self._composable_is_main:
            raise ValueError('this module does not accept extra args or kwargs')
        first_args, first_kwargs = get_composed_args(self.first, args, kwargs, tag_kwargs, None)
        second_args, second_kwargs = get_composed_args(self.second, args, kwargs, tag_kwargs, None)
        first_state = self.first.initial_state(batch_size, *first_args, **first_kwargs)
        return self.second.initial_composed_state(self.first, first_state, *second_args, **second_kwargs)

def _list_modules(u: Unidirectional) -> Iterable[Unidirectional]:
    result = []
    while isinstance(u, ComposedUnidirectional):
        result.append(u.second)
        u = u.first
    result.append(u)
    return reversed(result)

def get_composed_args(
    module: Unidirectional,
    args: list[Any],
    kwargs: dict[str, Any],
    tag_kwargs: dict[str, dict[str, Any]] | None,
    include_first: bool | None
) -> tuple[list[Any], dict[str, Any]]:
    new_args = []
    new_kwargs = dict(include_first=False) if include_first is not None else {}
    if module._composable_is_main:
        new_args.extend(args)
        new_kwargs.update(kwargs)
        if include_first is not None:
            new_kwargs['include_first'] = include_first
    if tag_kwargs:
        if isinstance(module, ComposedUnidirectional):
            new_kwargs['tag_kwargs'] = tag_kwargs
        else:
            for tag in module._composable_tags:
                these_tag_kwargs = tag_kwargs.get(tag)
                if these_tag_kwargs is not None:
                    new_kwargs.update(these_tag_kwargs)
    return new_args, new_kwargs
