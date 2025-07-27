from typing import Any

import torch
from torch_semiring_einsum import AutomaticBlockSize

from rau.unidirectional import ForwardResult
from rau.models.stack_nn.differentiable_stacks.semiring import log
from rau.models.stack_nn.differentiable_stacks.stack import DifferentiableStack
from rau.models.stack_nn.differentiable_stacks.vector_nondeterministic import (
    VectorNondeterministicStack
)
from rau.models.stack_nn.differentiable_stacks.nondeterministic import logits_to_actions
from .stack_attention import StackAttention

class NondeterministicStackAttention(StackAttention):

    def __init__(self,
        d_model: int,
        num_states: int,
        stack_alphabet_size: int,
        stack_embedding_size: int
    ) -> None:
        Q = num_states
        S = stack_alphabet_size
        super().__init__(
            d_model,
            num_actions=Q*S*Q*(S+S+1),
            pushed_vector_size=stack_embedding_size,
            stack_reading_size=Q*S*stack_embedding_size
        )
        self.num_states = num_states
        self.stack_alphabet_size = stack_alphabet_size
        self.stack_embedding_size = stack_embedding_size
        self.bottom_vector_logits = torch.nn.Parameter(torch.zeros(stack_embedding_size))

    def forward(self,
        input_sequence: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor | ForwardResult:
        return super().forward(
            input_sequence,
            sequence_length=input_sequence.size(1),
            *args,
            **kwargs
        )

    def initial_stack(self,
        batch_size: int,
        *args: Any,
        sequence_length: int | None = None,
        block_size: int | AutomaticBlockSize,
        **kwargs: Any
    ) -> DifferentiableStack:
        tensor = next(self.parameters())
        return VectorNondeterministicStack.new_empty(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            stack_embedding_size=self.stack_embedding_size,
            sequence_length=sequence_length,
            bottom_vector=torch.nn.functional.logsigmoid(self.bottom_vector_logits),
            block_size=block_size,
            dtype=tensor.dtype,
            device=tensor.device,
            semiring=log
        )

    def next_stack(self,
        stack: DifferentiableStack,
        action_tensor: torch.Tensor,
        pushed_vector: torch.Tensor
    ) -> DifferentiableStack:
        push, repl, pop = logits_to_actions(
            action_tensor,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            normalize=False
        )
        stack.update(push, repl, pop, pushed_vector)
        return stack
