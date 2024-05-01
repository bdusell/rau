from collections.abc import Callable, Iterable

import torch

def group_into_batches(
    examples: list[torch.Tensor],
    is_small_enough: Callable[[int, int], bool]
) -> Iterable[list[tuple[torch.Tensor, torch.Tensor]]]:
    examples.sort(key=len)
    batch = []
    for example in examples:
        batch.append(example)
        batch_size = len(batch)
        # Since the sequences are sorted in increasing order of length, the
        # length of the current sequence is the maximum length in the batch.
        max_length = len(example)
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
