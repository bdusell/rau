from collections.abc import Callable, Iterable
from typing import TypeVar

import torch

Example = TypeVar('Example')

def group_into_batches(
    examples: list[Example],
    is_small_enough: Callable[[int, int], bool],
    get_length: Callable[[Example], int] = len
) -> Iterable[list[Example]]:
    examples.sort(key=get_length)
    batch = []
    for example in examples:
        batch.append(example)
        batch_size = len(batch)
        # Since the sequences are sorted in increasing order of length, the
        # length of the current sequence is the maximum length in the batch.
        max_length = get_length(example)
        # A newly started batch always has size 1, and it should never be
        # discarded.
        if (
            batch_size > 1 and
            not is_small_enough(batch_size, max_length)
        ):
            batch.pop()
            if batch:
                yield batch
                batch = [example]
    if batch:
        yield batch
