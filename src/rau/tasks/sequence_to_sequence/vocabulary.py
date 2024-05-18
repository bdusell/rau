import dataclasses
import pathlib
from typing import Optional

import torch

from rau.vocab import Vocabulary, VocabularyBuilder, ToStringVocabularyBuilder
from rau.tasks.language_modeling.vocabulary import build_softmax_vocab

@dataclasses.dataclass
class SharedVocabularyData:
    tokens_in_target: list[str]
    tokens_only_in_source: list[str]
    allow_unk: bool

def load_shared_vocabulary_data_from_file(path: pathlib.Path) -> SharedVocabularyData:
    data = torch.load(path)
    return SharedVocabularyData(
        data['tokens_in_target'],
        data['tokens_only_in_source'],
        data['allow_unk']
    )

def get_vocabularies(
    vocabulary_data: SharedVocabularyData,
    builder: Optional[VocabularyBuilder]=None
) -> tuple[Vocabulary, Vocabulary, Vocabulary]:
    if builder is None:
        builder = ToStringVocabularyBuilder()
    softmax_vocab = build_softmax_vocab(
        vocabulary_data.tokens_in_target,
        vocabulary_data.allow_unk,
        builder
    )
    embedding_vocab = build_embedding_vocab(
        vocabulary_data.tokens_only_in_source,
        softmax_vocab,
        builder
    )
    return embedding_vocab, embedding_vocab, softmax_vocab

def build_embedding_vocab(tokens_only_in_source, softmax_vocab, builder):
    return (
        softmax_vocab +
        builder.content(tokens_only_in_source) +
        builder.reserved(['bos'])
    )
