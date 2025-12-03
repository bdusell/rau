import dataclasses

import torch

@dataclasses.dataclass
class PerUpdateLRSchedulerFunction:
    counter: int

    def __call__(self) -> float:
        raise NotImplementedError

class PerUpdateLRScheduler(torch.optim.lr_scheduler.LambdaLR):

    function: PerUpdateLRSchedulerFunction

    def __init__(self,
        optimizer: torch.optim.Optimizer,
        function: PerUpdateLRSchedulerFunction
    ) -> None:
        super().__init__(optimizer, function)
        self.function = function

    @property
    def counter(self) -> int:
        return self.function.counter

    @counter.setter
    def counter(self, value: int) -> None:
        self.function.counter = value
