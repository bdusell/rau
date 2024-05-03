from typing import Optional

import torch

def get_shared_embeddings(
    tie_embeddings: bool,
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    embedding_size: int,
    use_padding: bool
) -> Optional[torch.Tensor]:
    if tie_embeddings:
        return construct_shared_embeddings(
            input_vocabulary_size,
            output_vocabulary_size,
            embedding_size,
            use_padding
        )
    else:
        return None

def construct_shared_embeddings(
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    embedding_size: int,
    use_padding: bool
) -> torch.Tensor:
    if output_vocabulary_size > input_vocabulary_size:
        raise ValueError
    vocab_size = input_vocabulary_size + int(use_padding)
    return torch.nn.Parameter(torch.zeros(vocab_size, embedding_size))
