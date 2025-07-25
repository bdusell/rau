import dataclasses

import torch

from rau.unidirectional import (
    Unidirectional,
    ResidualUnidirectional,
    PositionalUnidirectional,
    StatelessUnidirectional
)

class AdditivePositional(PositionalUnidirectional):

    def forward_from_position(self, input_sequence, position):
        indexes = torch.arange(
            position,
            position + input_sequence.size(1),
            device=input_sequence.device
        )
        return input_sequence + indexes[None, :, None]

    def forward_at_position(self, input_tensor, position):
        return input_tensor + position

    def initial_state(self, *args, alpha, beta, **kwargs):
        assert alpha == 123
        assert beta == 'moo'
        return super().initial_state(*args, **kwargs)

def test_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    generator = torch.manual_seed(123)
    alpha = 123
    beta = 'moo'
    wrapped_model = AdditivePositional()
    model = ResidualUnidirectional(wrapped_model)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    expected_forward_output = (
        input_sequence +
        wrapped_model(input_sequence, include_first=False, alpha=alpha, beta=beta)
    )
    assert expected_forward_output.size() == (batch_size, sequence_length, input_size)
    forward_output = model(input_sequence, include_first=False, alpha=alpha, beta=beta)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    torch.testing.assert_close(forward_output, expected_forward_output)
    state = model.initial_state(batch_size, alpha=alpha, beta=beta)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        torch.testing.assert_close(output, forward_output[:, i])

class CountingStateless(StatelessUnidirectional):

    def __init__(self):
        super().__init__()
        self.num_forward_single_calls = 0
        self.num_forward_sequence_calls = 0

    def forward_single(self, input_tensor):
        self.num_forward_single_calls += 1
        return input_tensor

    def forward_sequence(self, input_sequence):
        self.num_forward_sequence_calls += 1
        return input_sequence

class CountingStateful(Unidirectional):

    def __init__(self):
        super().__init__()
        self.num_next_calls = 0
        self.num_output_calls = 0

    def initial_state(self, batch_size):
        return self.State(
            parent=self,
            input_tensor=None,
            _batch_size=batch_size
        )

    @dataclasses.dataclass
    class State(Unidirectional.State):

        parent: 'CountingStateful'
        input_tensor: torch.Tensor | None
        _batch_size: int | None

        def next(self, input_tensor):
            self.parent.num_next_calls += 1
            return dataclasses.replace(self, input_tensor=input_tensor)

        def output(self):
            self.parent.num_output_calls += 1
            if self.input_tensor is None:
                return torch.zeros(self._batch_size)
            else:
                return self.input_tensor

        def batch_size(self):
            return self._batch_size

def test_stateless_composed_residual():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    generator = torch.manual_seed(123)
    wrapped_model = CountingStateless()
    residual_model = ResidualUnidirectional(wrapped_model)
    input_model = CountingStateful()
    model = input_model.main() | residual_model
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        state = state.next(input_sequence[:, i])
    state.output()
    assert input_model.num_next_calls == sequence_length
    assert input_model.num_output_calls == 1
    assert wrapped_model.num_forward_single_calls == 1
    assert wrapped_model.num_forward_sequence_calls == 0

def test_stateful_composed_residual():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    generator = torch.manual_seed(123)
    wrapped_model = CountingStateful()
    residual_model = ResidualUnidirectional(wrapped_model)
    input_model = CountingStateful()
    model = input_model.main() | residual_model
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        state = state.next(input_sequence[:, i])
    state.output()
    assert input_model.num_next_calls == sequence_length
    assert input_model.num_output_calls == sequence_length
    assert wrapped_model.num_next_calls == sequence_length
    assert wrapped_model.num_output_calls == 1
