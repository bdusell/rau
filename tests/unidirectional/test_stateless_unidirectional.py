import torch

from rau.unidirectional import (
    Unidirectional,
    StatelessUnidirectional,
    StatelessLayerUnidirectional
)

class MyStateless(StatelessUnidirectional):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward_single(self, input_tensor):
        c = torch.sum(input_tensor, dim=1, keepdim=True)
        return input_tensor + c

    def forward_sequence(self, input_sequence):
        c = torch.sum(input_sequence, dim=2, keepdim=True)
        return input_sequence + c

    def initial_output(self, batch_size):
        return torch.ones((batch_size, self.input_size))

def test_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    model = MyStateless(input_size)
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, input_size)
        torch.testing.assert_close(output, forward_output[:, i])

def test_custom_args():
    batch_size = 5
    sequence_length = 13
    input_size = 7
    def func(x, alpha, *, beta):
        return x * alpha + beta
    model = StatelessLayerUnidirectional(func)
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, 123.0, include_first=False, beta=456.0)
    assert forward_output.size() == (batch_size, sequence_length, input_size)
    state = model.initial_state(batch_size, 123.0, beta=456.0)
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

    def forward_single(self, input_tensor):
        self.num_forward_single_calls += 1
        return input_tensor

class CountingStateful(Unidirectional):

    def __init__(self):
        super().__init__()
        self.num_next_calls = 0
        self.num_output_calls = 0

    def initial_state(self, batch_size):
        return self.State(self, None)

    class State(Unidirectional.State):

        def __init__(self, parent, input_tensor):
            super().__init__()
            self.parent = parent
            self.input_tensor = input_tensor

        def next(self, input_tensor):
            self.parent.num_next_calls += 1
            return self.parent.State(self.parent, input_tensor)

        def output(self):
            assert self.input_tensor is not None
            self.parent.num_output_calls += 1
            return self.input_tensor

def test_lazy_outputs_with_next():
    A = CountingStateless()
    B = CountingStateless()
    C = CountingStateful()
    D = CountingStateless()
    E = CountingStateless()
    M = A | B | C | D | E
    batch_size = 3
    sequence_length = 13
    model_size = 5
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, model_size), generator=generator)
    state = M.initial_state(batch_size)
    for input_tensor in input_sequence.transpose(0, 1):
        state = state.next(input_tensor)
    state.output()
    assert A.num_forward_single_calls == sequence_length
    assert B.num_forward_single_calls == sequence_length
    assert C.num_next_calls == sequence_length
    assert C.num_output_calls == 1
    assert D.num_forward_single_calls == 1
    assert E.num_forward_single_calls == 1

def test_lazy_outputs_with_fastforward():
    A = CountingStateless()
    B = CountingStateless()
    C = CountingStateful()
    D = CountingStateless()
    E = CountingStateless()
    M = A | B | C | D | E
    batch_size = 3
    sequence_length = 13
    model_size = 5
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, model_size), generator=generator)
    M.initial_state(batch_size).fastforward(input_sequence).output()
    assert A.num_forward_single_calls == sequence_length
    assert B.num_forward_single_calls == sequence_length
    assert C.num_next_calls == sequence_length
    assert C.num_output_calls == 1
    assert D.num_forward_single_calls == 1
    assert E.num_forward_single_calls == 1

def test_lazy_outputs_with_abnormal_composition_order():
    A = CountingStateless()
    B = CountingStateless()
    C = CountingStateful()
    D = CountingStateless()
    E = CountingStateless()
    M = A | B | (C | (D | E))
    batch_size = 3
    sequence_length = 13
    model_size = 5
    generator = torch.manual_seed(123)
    input_sequence = torch.rand((batch_size, sequence_length, model_size), generator=generator)
    state = M.initial_state(batch_size)
    for input_tensor in input_sequence.transpose(0, 1):
        state = state.next(input_tensor)
    state.output()
    assert A.num_forward_single_calls == sequence_length
    assert B.num_forward_single_calls == sequence_length
    assert C.num_next_calls == sequence_length
    assert C.num_output_calls == 1
    assert D.num_forward_single_calls == 1
    assert E.num_forward_single_calls == 1
