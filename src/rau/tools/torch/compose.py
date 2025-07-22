import itertools
from collections.abc import Iterable
from typing import Any

import torch

class BasicComposable(torch.nn.Module):
    r"""Base class for composable modules."""

    def __init__(self, main: bool, tags: Iterable[str] | None) -> None:
        super().__init__()
        self._composable_is_main = main
        self._composable_tags = dict.fromkeys(tags) if tags is not None else {}

    def __or__(self, other: torch.nn.Module) -> 'Composed':
        r"""Compose this module with another.

        :param other: Another module that will receive the outputs of this
            module as inputs and whose outputs will be returned by the resulting
            module.
        :return: A new module that feeds its inputs to this module, then feeds
            the outputs as inputs to ``other``, then returns the outputs of
            ``other``.
        """
        if not isinstance(other, BasicComposable):
            other = Composable(other)
        return Composed(self, other)

class Composable(BasicComposable):
    r"""A class that can be used to wrap any :py:class:`~torch.nn.Module` so
    that it can be used in a pipeline of :py:class:`~BasicComposable`\ s.
    """

    def __init__(self,
        module: torch.nn.Module,
        main: bool = False,
        tags: Iterable[str] = None,
        kwargs: dict[str, Any] | None = None
    ) -> None:
        r"""
        :param module: The module to wrapped. The new module will have the same
            inputs and outputs as this module.
        :param main: Whether this should be considered the main module, i.e., it
            should receive extra arguments from :py:meth:`Composed.forward` of
            a :py:class:`Composed` that contains it.
        :param kwargs: Optional keyword arguments that will be bound to the
            :py:meth:`forward` method.
        :param tags: Tags to assign to this module for argument routing.
        """
        super().__init__(main, tags)
        self.module = module
        self._composable_kwargs = kwargs if kwargs is not None else {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Same as the wrapped module.

        Automatically applies any bound keyword arguments.
        """
        return self.module(*args, **self._composable_kwargs, **kwargs)

    def main(self) -> 'Composable':
        r"""Mark this module as main.

        :return: Self.
        """
        self._composable_is_main = True
        return self

    def tag(self, tag: str) -> 'Composable':
        r"""Add a tag to this module for argument routing.

        :param tag: Tag name.
        :return: Self.
        """
        self._composable_tags[tag] = None
        return self

    def kwargs(self, **kwargs: Any) -> 'Composable':
        r"""Bind keyword arguments to be passed to
        :py:meth:`~torch.nn.Module.forward` of the wrapped module.

        :return: Self.
        """
        self._composable_kwargs.update(kwargs)
        return self

class Composed(BasicComposable):
    r"""A composition of two modules."""

    first: BasicComposable
    second: BasicComposable

    def __init__(self, first: BasicComposable, second: BasicComposable) -> None:
        super().__init__(
            first._composable_is_main or second._composable_is_main,
            itertools.chain(
                first._composable_tags.keys(),
                second._composable_tags.keys()
            )
        )
        self.first = first
        self.second = second

    def forward(self,
        x: Any,
        *args: Any,
        tag_kwargs: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Any:
        r"""Feed the input ``x`` as input to the first module, feed the outputs
        as inputs to the second module, and return the output of the second
        module.

        :param x: The input to the first module.
        :param args: Extra arguments that will be passed to the main module.
        :param tag_kwargs: A dict mapping tag names to dicts of keyword
            arguments. These keyword arguments will be passed only to modules
            with the corresponding tags.
        :param kwargs: Extra keyword arguments that will be passed to the main
            module.
        """
        first_args, first_kwargs = get_composed_args(self.first, args, kwargs, tag_kwargs)
        second_args, second_kwargs = get_composed_args(self.second, args, kwargs, tag_kwargs)
        return self.second(self.first(x, *first_args, **first_kwargs), *second_args, **second_kwargs)

def get_composed_args(
    module: BasicComposable,
    args: list[Any],
    kwargs: dict[str, Any],
    tag_kwargs: dict[str, dict[str, Any]] | None
) -> tuple[list[Any], dict[str, Any]]:
    new_args = []
    new_kwargs = {}
    if module._composable_is_main:
        new_args.extend(args)
        new_kwargs.update(kwargs)
    if tag_kwargs:
        if isinstance(module, Composed):
            new_kwargs['tag_kwargs'] = tag_kwargs
        else:
            for tag in module._composable_tags:
                these_tag_kwargs = tag_kwargs.get(tag)
                if these_tag_kwargs is not None:
                    new_kwargs.update(these_tag_kwargs)
    return new_args, new_kwargs
