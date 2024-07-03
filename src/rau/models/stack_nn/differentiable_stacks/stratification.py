import torch

from .stack import DifferentiableStack

class StratificationStack(DifferentiableStack):

    def __init__(self, elements, bottom):
        self.elements = elements
        self.bottom = bottom

    @staticmethod
    def new_empty(
        batch_size: int,
        stack_embedding_size: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> 'StratificationStack':
        return StratificationStack(
            elements=[],
            bottom=torch.zeros((batch_size, stack_embedding_size), dtype=dtype, device=device)
        )

    def reading(self):
        device = self.bottom.device
        batch_size = self.bottom.size(0)
        result = self.bottom
        strength_left = torch.ones((batch_size,), device=device)
        for value, strength in reversed(self.elements):
            result = result + value * torch.min(
                strength,
                torch.nn.functional.relu(strength_left)
            )[:, None]
            strength_left = strength_left - strength
        return result

    def next(self, actions, push_value):
        return StratificationStack(
            self.next_elements(actions, push_value),
            self.bottom
        )

    def next_elements(self, actions, push_value):
        push_strength = actions[:, 0]
        pop_strength = actions[:, 1]
        result = []
        strength_left = pop_strength
        for value, strength in reversed(self.elements):
            result.append((
                value,
                torch.nn.functional.relu(
                    strength -
                    torch.nn.functional.relu(strength_left)
                )
            ))
            strength_left = strength_left - strength
        result.reverse()
        result.append((push_value, push_strength))
        return result

    def transform_tensors(self, func):
        return StratificationStack(
            func(self.elements),
            func(self.bottom)
        )

    def batch_size(self):
        return self.elements.size(0)
