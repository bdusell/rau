from typing import Any

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tasks.common.training_loop import (
    add_training_loop_arguments as common_add_training_loop_arguments,
    get_training_loop_kwargs,
    TrainingLoop
)
from rau.tasks.language_modeling.training_loop import (
    get_cross_entropy_loss,
    evaluate_batch
)
from .batching import group_into_batches
from .data import VocabularyContainer
from .model import ModelSourceAndTarget

def add_training_loop_arguments(parser):
    common_add_training_loop_arguments(
        parser,
        max_tokens_per_batch_help=
        'The maximum number of tokens allowed per batch. This puts a limit on '
        'the number of elements included in a single source or target batch '
        'tensor, including BOS, EOS, and padding tokens. If a single example '
        'exceeds the limit, it is not discarded, but included in a batch by '
        'itself.'
    )

Example = tuple[torch.Tensor, torch.Tensor]
PreparedBatch = tuple[ModelSourceAndTarget, torch.Tensor]

class SequenceToSequenceTrainingLoop(TrainingLoop[
    Example,
    PreparedBatch,
    VocabularyContainer
]):

    def get_validation_metric_name(self):
        return 'cross_entropy_per_token'

    def get_validation_metric_mode(self):
        return 'min'

    def generate_batches(self, examples, max_tokens):
        return generate_batches(examples, max_tokens)

    def get_prepared_batch_info(self, prepared_batch):
        model_input, correct_target = prepared_batch
        return dict(
            source_input_size=tuple(model_input.source.size()),
            target_input_size=tuple(model_input.target.size()),
            target_output_size=tuple(correct_target.size())
        )

    def log_failed_batch(self, vocabulary, batch, info, console_logger, event_logger):
        if info is not None:
            console_logger.info(f'  source input size: {info.get("source_input_size")}')
            console_logger.info(f'  target input size: {info.get("target_input_size")}')
            console_logger.info(f'  target output size: {info.get("target_output_size")}')
        source_tokens = sum(len(s) for s, t in batch)
        console_logger.info(f'  source tokens: {source_tokens}')
        target_tokens = sum(len(t) for s, t in batch)
        console_logger.info(f'  target tokens: {target_tokens}')
        lengths = [(len(s), len(t)) for s, t in batch]
        console_logger.info(f'  sequence lengths: {lengths}')
        token_strs = [
            (
                [vocabulary.source_vocab.to_string(w) for w in s],
                [vocabulary.target_output_vocab.to_string(w) for w in t]
            )
            for s, t in batch
        ]
        sequences_str = '\n'.join(f'{" ".join(s)}\t{" ".join(t)}' for s, t in token_strs)
        console_logger.info(f'  sequences:\n{sequences_str}')
        return dict(
            **(info or {}),
            examples=token_strs
        )

    def get_loss(self, model, model_interface, prepared_batch):
        return get_cross_entropy_loss(self, model, model_interface, prepared_batch)

    def evaluate_batch(self, model, model_interface, prepared_batch):
        return evaluate_batch(model, model_interface, prepared_batch)

def generate_batches(examples, max_tokens):
    return group_into_batches(
        examples,
        lambda b, s, t: b * s <= max_tokens and b * t <= max_tokens
    )
