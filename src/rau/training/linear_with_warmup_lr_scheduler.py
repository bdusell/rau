import dataclasses

import torch

from .per_update_lr_scheduler import PerUpdateLRScheduler, PerUpdateLRSchedulerFunction

@dataclasses.dataclass
class LinearWithWarmupFunction(PerUpdateLRSchedulerFunction):
    warmup: int
    total: int

    def __call__(self, epoch: int) -> float:
        return get_linear_with_warmup_ratio(self.warmup, self.total, self.counter)

class LinearWithWarmupLRScheduler(PerUpdateLRScheduler):

    def __init__(self,
        optimizer: torch.optim.Optimizer,
        warmup: int,
        total: int
    ) -> None:
        super().__init__(optimizer, LinearWithWarmupFunction(0, warmup, total))

def get_linear_with_warmup_ratio(warmup: int, total: int, counter: int) -> float:
    if not (0 <= counter <= total and 0 <= warmup <= total):
        raise ValueError(
            f'invalid values for linear with warmup schedule: '
            f'warmup = {warmup}, total = {total}, counter = {counter}'
        )
    if counter <= warmup:
        return (counter + 1) / (warmup + 1)
    else:
        return (total - counter + 1) / (total - warmup + 1)
