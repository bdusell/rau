import torch

from rau.tools.torch import Composable, Composed

def test_two_layers():
    generator = torch.manual_seed(123)
    A = torch.nn.Linear(3, 7)
    B = torch.nn.Linear(7, 5)
    for p in (*A.parameters(), *B.parameters()):
        p.data.uniform_(generator=generator)
    M = Composable(A) | Composable(B)
    assert isinstance(M, Composed)
    x = torch.rand((11, 3), generator=generator)
    y = M(x)
    assert y.size() == (11, 5)
    expected_y = B(A(x))
    torch.testing.assert_close(y, expected_y)

class ModuleA(torch.nn.Module):

    def forward(self, x, foo):
        assert foo == 1
        return x

class ModuleB(torch.nn.Module):

    def forward(self, x, bar, baz):
        assert bar == 2
        assert baz == 3
        return x

class ModuleMain(torch.nn.Module):

    def forward(self, x, arg, hello):
        assert arg == 42
        assert hello == 'world'
        return x

def test_argument_routing():
    generator = torch.manual_seed(123)
    A = ModuleA()
    B = ModuleB()
    C = ModuleMain()
    M = Composable(A).tag('a') | Composable(C).main() | Composable(B).tag('b')
    x = torch.rand((11, 3), generator=generator)
    y = M(
        x,
        42,
        hello='world',
        tag_kwargs=dict(
            a=dict(foo=1),
            b=dict(bar=2, baz=3)
        )
    )

def test_kwarg_binding():
    generator = torch.manual_seed(123)
    A = ModuleA()
    B = ModuleB()
    C = ModuleMain()
    M = (
        Composable(A).kwargs(foo=1) |
        Composable(C).main().kwargs(hello='world') |
        Composable(B).tag('b').kwargs(baz=3)
    )
    x = torch.rand((11, 3), generator=generator)
    y = M(
        x,
        42,
        tag_kwargs=dict(
            b=dict(bar=2)
        )
    )
