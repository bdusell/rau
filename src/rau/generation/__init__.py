r"""This module includes algorithms for sampling strings of symbols from
autoregressive language models or decoders which are encapsulated in
:py:class:`~rau.unidirectional.Unidirectional.State`\ s.
"""

from .sample import sample
from .greedy import decode_greedily
from .beam_search import beam_search
