import re

STACK_RE = re.compile(r'^stratification-(\d+)|superposition-(\d+)|nondeterministic-(\d+)-(\d+)|vector-nondeterministic-(\d+)-(\d+)-(\d+)$')

StackRNNStack = tuple[str, tuple]

def parse_stack_rnn_stack(s: str) -> StackRNNStack:
    m = STACK_RE.match(s)
    if m is None:
        raise ValueError(f'invalid stack RNN stack string: {s}')
    strat_m, sup_m, nd_q, nd_s, vnd_q, vnd_s, vnd_m = m.groups()
    if strat_m is not None:
        return 'stratification', (int(strat_m),)
    elif sup_m is not None:
        return 'superposition', (int(sup_m),)
    elif nd_q is not None:
        return 'nondeterministic', (int(nd_q), int(nd_s))
    elif vnd_q is not None:
        return 'vector-nondeterministic', (int(vnd_q), int(vnd_s), int(vnd_m))
