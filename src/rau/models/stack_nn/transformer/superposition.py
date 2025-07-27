import math
from typing import Any, Literal

import torch

from rau.unidirectional import ForwardResult
from rau.models.stack_nn.differentiable_stacks.stack import DifferentiableStack
from rau.models.stack_nn.differentiable_stacks.superposition import SuperpositionStack
from .stack_attention import StackAttention

class SuperpositionStackAttention(StackAttention):

    def __init__(self, d_model: int, stack_embedding_size: int) -> None:
        super().__init__(
            d_model=d_model,
            num_actions=3,
            pushed_vector_size=stack_embedding_size,
            stack_reading_size=stack_embedding_size
        )
        self.stack_embedding_size = stack_embedding_size

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
        sequence_length: int | Literal[math.inf] = math.inf,
        **kwargs: Any
    ) -> DifferentiableStack:
        t = next(self.parameters())
        return SuperpositionStack.new_empty(
            batch_size=batch_size,
            stack_embedding_size=self.stack_embedding_size,
            # NOTE The +1 is necessary because, unlike an RNN, the last stack
            # reading is actually used.
            max_sequence_length=sequence_length + 1,
            max_depth=math.inf,
            dtype=t.dtype,
            device=t.device
        )

    def next_stack(self,
        stack: DifferentiableStack,
        action_tensor: torch.Tensor,
        pushed_vector: torch.Tensor
    ) -> DifferentiableStack:
        # action_tensor : batch_size x 3
        d_model = pushed_vector.size(1)
        # push_prob, etc. : batch_size x d_model
        push_prob, pop_prob, noop_prob = (
            # prob : batch_size
            prob[:, None].expand(-1, d_model)
            for prob in torch.unbind(action_tensor, dim=1)
        )
        return stack.next(push_prob, pop_prob, noop_prob, pushed_vector)

    def transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(actions, dim=-1)
