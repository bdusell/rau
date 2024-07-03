import math
from typing import Literal

import torch

from rau.tools.torch.layer import Layer, MultiLayer

from .stack import DifferentiableStack

class SuperpositionStack(DifferentiableStack):

    def __init__(self, elements, timestep, max_sequence_length, max_depth):
        # elements : batch_size x stack_embedding_size x stack_height
        self.elements = elements
        self.timestep = timestep
        self.max_sequence_length = max_sequence_length
        self.max_depth = max_depth

    @staticmethod
    def new_empty(
        batch_size: int,
        stack_embedding_size: int,
        max_sequence_length: int,
        max_depth: int | Literal[math.inf],
        dtype: torch.dtype,
        device: torch.device
    ) -> 'SuperpositionStack':
        return SuperpositionStack(
            elements=torch.zeros((batch_size, stack_embedding_size, 0), dtype=dtype, device=device),
            timestep=0,
            max_sequence_length=max_sequence_length,
            max_depth=max_depth
        )

    def reading(self):
        batch_size, reading_size, num_elements = self.elements.size()
        if num_elements > 0:
            return self.elements[:, :, 0]
        else:
            return torch.zeros(batch_size, reading_size, device=self.elements.device)

    def next(self, push_prob, pop_prob, noop_prob, push_value):
        return SuperpositionStack(
            self.next_elements(push_prob, pop_prob, noop_prob, push_value),
            self.timestep + 1,
            self.max_sequence_length,
            self.max_depth
        )

    def next_elements(self, push_prob, pop_prob, noop_prob, push_value):
        # push_prob : batch_size x stack_embedding_size
        # pop_prob : batch_size x stack_embedding_size
        # noop_prob : batch_size x stack_embedding_size
        # push_value : batch_size x stack_embedding_size
        # self.elements : batch_size x stack_embedding_size x stack_height
        batch_size = self.elements.size(0)
        device = self.elements.device
        next_timestep = self.timestep + 1
        actual_stack_height = min(
            next_timestep,
            self.max_sequence_length - next_timestep,
            self.max_depth
        )
        max_push_elements = actual_stack_height - 1
        push_elements = self.elements
        if push_elements.size(2) > max_push_elements:
            push_elements = push_elements[:, :, :max_push_elements]
        push_terms = push_prob[:, :, None] * torch.cat([
            push_value[:, :, None],
            push_elements
        ], dim=2)
        # push_terms : batch_size x stack_embedding_size x stack_height
        pop_terms = pop_prob[:, :, None] * self.elements[:, :, 1:1+actual_stack_height]
        # pop_terms : batch_size x stack_embedding_size x stack_height
        noop_terms = noop_prob[:, :, None] * self.elements[:, :, :actual_stack_height]
        # noop_terms : batch_size x stack_embedding_size x stack_height
        return jagged_sum(jagged_sum(push_terms, noop_terms), pop_terms)

    def detach(self):
        return self.transform_tensors(lambda x: x.detach())

    def slice_batch(self, s):
        return self.transform_tensors(lambda x: x[s, ...])

    def transform_tensors(self, func):
        return SuperpositionStack(
            func(self.elements),
            self.timestep,
            self.max_sequence_length,
            self.max_depth
        )

    def batch_size(self):
        return self.elements.size(0)

def jagged_sum(a, b):
    # Efficiently adds two stack tensors which may not have the same number
    # of stack elements.
    # Precondition: a.size(2) >= b.size(2)
    a_size = a.size(2)
    b_size = b.size(2)
    if b_size == 0:
        # This branch is needed because .backward() throws an exception
        # for some reason when b_size is 0.
        return a
    elif a_size == b_size:
        return a + b
    else:
        return torch.cat([
            a[:, :, :b_size] + b,
            a[:, :, b_size:]
        ], dim=2)
