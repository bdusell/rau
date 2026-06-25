from collections.abc import Callable, Iterable

import torch

def group_into_batches[T](
    examples: list[T],
    is_small_enough: Callable[[int, int], bool],
    get_length: Callable[[T], int]
) -> Iterable[list[T]]:
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

def group_into_same_length_batches[T](
    examples: list[T],
    is_small_enough: Callable[[int, int], bool],
    get_length: Callable[[T], int]
) -> Iterable[list[T]]:
    examples.sort(key=get_length)
    batch: list[T] = []
    for example in examples:
        if batch and get_length(example) != get_length(batch[0]):
            yield batch
            batch = [example]
        else:
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
