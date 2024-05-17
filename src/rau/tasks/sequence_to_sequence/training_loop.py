from typing import Any

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tasks.common.training_loop import (
    add_training_loop_arguments as common_add_training_loop_arguments,
    get_training_loop_kwargs,
    TrainingLoop
)
from rau.tasks.common.data import Dataset
from .batching import group_into_batches

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

class SequenceToSequenceTrainingLoop(TrainingLoop[Example]):

    def get_validation_metric_name(self):
        return 'cross_entropy_per_token'

    def get_validation_metric_mode(self):
        return 'min'

    def generate_batches(self, examples, max_tokens):
        return group_into_batches(
            examples,
            lambda b, s, t: b * s <= max_tokens and b * t <= max_tokens
        )

    def get_prepared_batch_info(self, prepared_batch):
        model_input, correct_target = prepared_batch
        return dict(
            source_input_size=tuple(model_input.source.size()),
            target_input_size=tuple(model_input.target.size()),
            target_output_size=tuple(correct_target.size())
        )

    def log_failed_batch(self, dataset, batch, info, console_logger, event_logger):
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
                [dataset.source_vocab.to_string(w) for w in s],
                [dataset.target_output_vocab.to_string(w) for w in t]
            )
            for s, t in batch
        ]
        sequences_str = '\n'.join(f'{" ".join(s)}\t{" ".join(t)}' for s, t in token_strs)
        console_logger.info(f'  sequences:\n{sequences_str}')
        return dict(
            **info,
            examples=token_strs
        )

    def get_loss(self, model, model_interface, dataset, prepared_batch):
        cross_entropy, num_symbols = self.get_cross_entropy(
            model,
            model_interface,
            dataset,
            prepared_batch,
            reduction='none',
            label_smoothing_factor=self.label_smoothing_factor
        )
        loss_numerators = torch.sum(cross_entropy, dim=1)
        return loss_numerators, num_symbols

    def evaluate_batch(self, model, model_interface, dataset, prepared_batch):
        cross_entropy, num_symbols = self.get_cross_entropy(
            model,
            model_interface,
            dataset,
            prepared_batch,
            reduction='sum',
            label_smoothing_factor=0.0
        )
        return {
            'cross_entropy_per_token' : (cross_entropy.item(), num_symbols)
        }

    def get_cross_entropy(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        dataset: Dataset[Example],
        prepared_batch: list[Example],
        reduction: str,
        label_smoothing_factor: float
    ) -> tuple[torch.Tensor, int, dict[str, Any]]:
        model_input, correct_target = prepared_batch
        # TODO Cache this lookup.
        pad_index = model_interface.get_output_padding_index(dataset)
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
