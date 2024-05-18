from typing import Any

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tasks.common.training_loop import (
    add_training_loop_arguments as common_add_training_loop_arguments,
    get_training_loop_kwargs,
    TrainingLoop
)
from .batching import group_into_batches
from .data import VocabularyContainer

def add_training_loop_arguments(parser):
    common_add_training_loop_arguments(parser,
        max_tokens_per_batch_help=
        'The maximum number of tokens allowed per batch. This puts a limit on '
        'the number of elements included in a single batch tensor, including '
        'BOS, EOS, and padding tokens. If a single example exceeds the limit, '
        'it is not discarded, but included in a batch by itself.'
    )

Example = torch.Tensor
PreparedBatch = tuple[torch.Tensor, torch.Tensor]

class LanguageModelingTrainingLoop(TrainingLoop[
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
            input_size=tuple(model_input.size()),
            output_size=tuple(correct_target.size())
        )

    def log_failed_batch(self, vocabulary, batch, info, console_logger, event_logger):
        if info is not None:
            console_logger.info(f'  input size: {info.get("input_size")}')
            console_logger.info(f'  output size: {info.get("output_size")}')
        tokens = sum(map(len, batch))
        console_logger.info(f'  tokens: {tokens}')
        lengths = list(map(len, batch))
        console_logger.info(f'  sequence lengths: {lengths}')
        token_strs = [
            [vocabulary.input_vocab.to_string(w) for w in x]
            for x in batch
        ]
        sequences_str = '\n'.join(' '.join(x) for x in token_strs)
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
    return group_into_batches(examples, lambda b, n: b * n <= max_tokens)

def get_cross_entropy_loss(training_loop, model, model_interface, prepared_batch):
    cross_entropy, num_symbols = get_cross_entropy(
        model,
        model_interface,
        prepared_batch,
        reduction='none',
        label_smoothing_factor=training_loop.label_smoothing_factor
    )
    loss_numerators = torch.sum(cross_entropy, dim=1)
    return loss_numerators, num_symbols

def evaluate_batch(model, model_interface, prepared_batch):
    cross_entropy, num_symbols = get_cross_entropy(
        model,
        model_interface,
        prepared_batch,
        reduction='sum',
        label_smoothing_factor=0.0
    )
    return {
        'cross_entropy_per_token' : (cross_entropy.item(), num_symbols)
    }

def get_cross_entropy(
    model: torch.nn.Module,
    model_interface: ModelInterface,
    prepared_batch: list[Example],
    reduction: str,
    label_smoothing_factor: float
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    model_input, correct_target = prepared_batch
    pad_index = model_interface.output_padding_index
    logits = model_interface.get_logits(model, model_input)
    cross_entropy = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1),
        correct_target,
        ignore_index=pad_index,
        reduction=reduction,
        label_smoothing=label_smoothing_factor
    )
    num_symbols = torch.sum(correct_target != pad_index).item()
    return cross_entropy, num_symbols
