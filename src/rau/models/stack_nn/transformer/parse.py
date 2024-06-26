import re

from .stack_attention import StackAttention
from .superposition import SuperpositionStackAttention
from .nondeterministic import NondeterministicStackAttention

STACK_TRANSFORMER_LAYERS_HELP_MESSAGE = (
    'This can be a mix of standard layers and stack attention layers, in any '
    'order. Layers are separated by `.` characters. A layer can be: '
    '(1) an integer n, which indicates n standard attention layers in a row '
    '(2) superposition-<m>, where <m> is an integer, indicating a '
    'superposition stack attention layer with stack embedding size <m> '
    '(3) nondeterministic-<q>-<s>-<m>, where <q>, <s>, <m> are integers, '
    'indicating a nondeterministic stack attention layer with <q> PDA states, '
    '<s> stack symbol types, and stack embedding size <m>.'
)

LAYER_RE = re.compile(r'^(\d+)|superposition-(\d+)|nondeterministic-(\d+)-(\d+)-(\d+)$')

StackTransformerLayers = list[tuple[str, tuple]]

def parse_stack_transformer_layers(s: str) -> StackTransformerLayers:
    return list(_parse_stack_transformer_layers_gen(s))

def _parse_stack_transformer_layers_gen(s):
    for part in s.split('.'):
        m = LAYER_RE.match(part)
        if m is None:
            raise ValueError(f'invalid stack transformer layer string: {part}')
        tf_layers, sup_m, nd_q, nd_s, nd_m = m.groups()
        if tf_layers is not None:
            yield 'transformer', (int(tf_layers),)
        elif sup_m is not None:
            yield 'superposition', (int(sup_m),)
        elif nd_q is not None:
            yield 'nondeterministic', (int(nd_q), int(nd_s), int(nd_m))

def get_stack_attention_func(layer_type: str, layer_args: tuple, d_model: int) -> StackAttention:
    if layer_type == 'superposition':
        stack_embedding_size, = layer_args
        return SuperpositionStackAttention(
            d_model=d_model,
            stack_embedding_size=stack_embedding_size
        )
    elif layer_type == 'nondeterministic':
        num_states, stack_alphabet_size, stack_embedding_size = layer_args
        return NondeterministicStackAttention(
            d_model=d_model,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            stack_embedding_size=stack_embedding_size
        )
    else:
        raise ValueError
