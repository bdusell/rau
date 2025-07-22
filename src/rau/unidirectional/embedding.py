import torch

from rau.tools.torch.embedding_layer import EmbeddingLayer
from .stateless import StatelessLayerUnidirectional

class EmbeddingUnidirectional(StatelessLayerUnidirectional):

    def __init__(self,
        vocabulary_size: int,
        output_size: int,
        use_padding: bool,
        shared_embeddings: torch.Tensor | None = None
    ):
        super().__init__(EmbeddingLayer(
            vocabulary_size,
            output_size,
            use_padding,
            shared_embeddings
        ))
