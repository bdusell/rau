import dataclasses
import random

import torch

def get_random_generator_and_seed(random_seed):
    random_seed = get_random_seed(random_seed)
    return random.Random(random_seed), random_seed

def get_random_seed(random_seed):
    return random.getrandbits(32) if random_seed is None else random_seed

def get_optimizer(saver, args):
    OptimizerClass = getattr(torch.optim, args.optimizer)
    return OptimizerClass(
        saver.model.parameters(),
        lr=args.learning_rate
    )

@dataclasses.dataclass
class ParameterUpdateResult:
    loss_numer: float
    num_symbols: int

class LossAccumulator:

    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0

    def update(self, numerator, denominator):
        self.numerator += numerator
        self.denominator += denominator

    def get_value(self):
        return self.numerator / self.denominator
