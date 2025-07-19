import torch

def get_shared_embeddings(
    tie_embeddings: bool,
    input_vocabulary_size: int,
    output_vocabulary_size: int,
    embedding_size: int,
    use_padding: bool
) -> torch.Tensor | None:
    r"""Construct a matrix of embedding vectors that can be used as tied input
    embeddings and output embeddings.

    The size of the output vocabulary must be no greater than the size of the
    input vocabulary.

    :param tie_embeddings: If false, ``None`` is returned, indicating that a
        shared embedding matrix should not be used.
    :param input_vocabulary_size: The size of the input vocabulary.
    :param output_vocabulary_size: The size of the output vocabulary.
    :param embedding_size: The size of the embedding vectors.
    :param use_padding: Whether to ensure that the embedding matrix is big
        enough to accommodate an index for a reserved padding symbol.
    :return: A matrix of size :math:`\text{input vocabulary size} \times
        \text{embedding size}`. If ``use_padding`` is true, then 1 will be added
        to input vocabulary size. If ``tie_embeddings`` is false, ``None`` is
        returned.
    """
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
        raise ValueError(
            f'output_vocabulary_size ({output_vocabulary_size}) cannot be '
            f'greater than input_vocabulary_size ({input_vocabulary_size}) '
            f'when using shared embeddings'
        )
    vocab_size = input_vocabulary_size + int(use_padding)
    return torch.nn.Parameter(torch.zeros(vocab_size, embedding_size))
