r"""This module includes algorithms for sampling strings of symbols from
autoregressive language models or decoders which are encapsulated in
:py:class:`~rau.unidirectional.Unidirectional.State`\ s.
"""

from .sample import sample
from .beam_search import beam_search
