import argparse
import dataclasses
import datetime
import logging
import random
from collections.abc import Iterable
from typing import Any, Generic, Literal, Optional, TypeVar

import humanfriendly
import torch

from rau.tools.logging import Logger
from rau.tools.ticker import TimedTicker
from rau.tools.torch.saver import ModelSaver
from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.profile import reset_memory_profiler, get_peak_memory
from rau.training.early_stopping import UpdatesWithoutImprovement
from .data import Dataset

Example = TypeVar('Example')
Batch = list[Example]

def add_training_loop_arguments(
    parser: argparse.ArgumentParser,
    max_tokens_per_batch_help: str
) -> None:
    group = parser.add_argument_group('Training options')
    group.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages during training.')
    group.add_argument('--max-epochs', type=int, required=True,
        help='The maximum number of epochs to run training for.')
    group.add_argument('--random-shuffling-seed', type=int,
        help='Random seed used for random shuffling of the training data.')
    group.add_argument('--max-tokens-per-batch', type=int, required=True,
        help=max_tokens_per_batch_help)
    group.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam',
        help='The algorithm to use for parameter optimization.')
    group.add_argument('--initial-learning-rate', type=float, required=True,
        help='The initial learning rate.')
    group.add_argument('--label-smoothing-factor', type=float, default=0.0,
        help='The label smoothing factor to use with the cross-entropy '
             'loss. Default is 0 (no label smoothing).')
    group.add_argument('--gradient-clipping-threshold', type=float,
        help='The threshold to use for L2 gradient clipping. If not given, '
             'gradients are never clipped.')
    group.add_argument('--early-stopping-patience', type=int, required=True,
        help='The allowed number of epochs of no improvement in performance '
             'on the validation data before training stops early.')
    group.add_argument('--learning-rate-patience', type=int, required=True,
        help='The allowed number of epochs of no improvement in performance '
             'on the validation data before the learning rate is reduced.')
    group.add_argument('--learning-rate-decay-factor', type=float, required=True,
        help='A value between 0 and 1 that the learning rate will be '
             'multiplied by whenever it should be decreased.')
    group.add_argument('--examples-per-checkpoint', type=int, required=True,
        help='An evaluation checkpoint will be run on the validation data '
             'every time this many training examples have been processed.')

def get_training_loop_kwargs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace
) -> dict[str, object]:
    result = {}
    result['show_progress'] = not args.no_progress
    for name in [
        'max_epochs',
        'random_shuffling_seed',
        'max_tokens_per_batch',
        'optimizer',
        'initial_learning_rate',
        'label_smoothing_factor',
        'gradient_clipping_threshold',
        'early_stopping_patience',
        'learning_rate_patience',
        'learning_rate_decay_factor',
        'examples_per_checkpoint'
    ]:
        result[name] = getattr(args, name)
    return result

@dataclasses.dataclass
class TrainingLoop(Generic[Example]):

    show_progress: bool
    max_epochs: int
    random_shuffling_seed: int
    max_tokens_per_batch: int
    optimizer: Literal['SGD', 'Adam']
    initial_learning_rate: float
    label_smoothing_factor: Optional[float]
    gradient_clipping_threshold: Optional[float]
    early_stopping_patience: int
    learning_rate_patience: int
    learning_rate_decay_factor: float
    examples_per_checkpoint: int

    def get_validation_metric_name(self) -> str:
        raise NotImplementedError

    def get_validation_metric_mode(self) -> Literal['min', 'max']:
        raise NotImplementedError

    def generate_batches(self,
        examples: Iterable[Example],
        max_tokens: int
    ) -> Iterable[Batch]:
        raise NotImplementedError

    def get_prepared_batch_info(self,
        prepared_batch: Any
    ) -> dict[str, Any]:
        raise NotImplementedError

    def log_failed_batch(self,
        dataset: Dataset[Example],
        batch: Batch,
        info: dict[str, Any],
        console_logger: logging.Logger,
        event_logger: Logger
    ) -> dict[str, Any]:
        raise NotImplementedError

    def get_loss(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        dataset: Dataset[Example],
        prepared_batch: Any
    ) -> tuple[torch.Tensor, int, dict[str, Any]]:
        raise NotImplementedError

    def evaluate_batch(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        dataset: Dataset[Example],
        prepared_batch: Any
    ) -> tuple[float, int]:
        raise NotImplementedError

    def run(self,
        saver: ModelSaver,
        model_interface: ModelInterface,
        dataset: Dataset[Example],
        console_logger: logging.Logger,
        event_logger: Logger
    ) -> None:
        """
        NOTE: When this function returns, the model's parameters will be those of
        the *last* epoch, not necessarily the *best* epoch.
        """
        device = model_interface.get_device(None)
        do_profile_memory = device.type == 'cuda'
        random_shuffling_generator, random_shuffling_seed = \
            get_random_generator_and_seed(self.random_shuffling_seed)
        console_logger.info(f'random shuffling seed: {random_shuffling_seed}')
        OptimizerClass = getattr(torch.optim, self.optimizer)
        optimizer = OptimizerClass(
            saver.model.parameters(),
            lr=self.initial_learning_rate
        )
        validation_metric = self.get_validation_metric_name()
        validation_metric_mode = self.get_validation_metric_mode()
        early_stopping = UpdatesWithoutImprovement(
            validation_metric_mode,
            patience=self.early_stopping_patience
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=validation_metric_mode,
            patience=self.learning_rate_patience,
            factor=self.learning_rate_decay_factor
        )
        console_logger.info(f'training examples: {len(dataset.training_data)}')
        num_validation_examples = len(dataset.validation_data)
        console_logger.info(f'validation examples: {num_validation_examples}')
        validation_batches = list(self.generate_batches(
            dataset.validation_data,
            self.max_tokens_per_batch
        ))
        console_logger.info(f'validation batches: {len(validation_batches)}')
        model_interface.on_before_process_pairs(
            saver,
            [dataset.training_data, dataset.validation_data]
        )
        dataset.validation_data = None
        event_logger.log('start_training', dict(
            num_training_examples=len(dataset.training_data),
            num_validation_examples=num_validation_examples,
            num_validation_batches=len(validation_batches),
            max_epochs=self.max_epochs,
            random_shuffling_seed=random_shuffling_seed,
            optimizer=self.optimizer,
            initial_learning_rate=self.initial_learning_rate,
            label_smoothing_factor=self.label_smoothing_factor,
            early_stopping_patience=self.early_stopping_patience,
            learning_rate_patience=self.learning_rate_patience,
            learning_rate_decay_factor=self.learning_rate_decay_factor,
            gradient_clipping_threshold=self.gradient_clipping_threshold,
            examples_per_checkpoint=self.examples_per_checkpoint
        ))
        epoch_no = 0
        examples_since_checkpoint = 0
        checkpoint_no = 0
        best_validation_scores = None
        best_checkpoint_no = None
        best_epoch_no = None
        total_start_time = datetime.datetime.now()
        for _ in range(self.max_epochs):
            epoch_start_time = datetime.datetime.now()
            console_logger.info(f'epoch #{epoch_no + 1}')
            random_shuffling_generator.shuffle(dataset.training_data)
            batches = list(self.generate_batches(
                dataset.training_data,
                self.max_tokens_per_batch
            ))
            random_shuffling_generator.shuffle(batches)
            epoch_loss = LossAccumulator()
            if self.show_progress:
                progress_loss = LossAccumulator()
                progress_num_examples = 0
                progress_start_time = datetime.datetime.now()
                ticker = TimedTicker(len(batches), 1)
            if do_profile_memory:
                reset_memory_profiler(device)
            should_stop = False
            for batch_no, batch in enumerate(batches):
                try:
                    loss_numerator, loss_denominator = self.run_parameter_update(
                        saver,
                        model_interface,
                        dataset,
                        optimizer,
                        batch
                    )
                    epoch_loss.update(loss_numerator, loss_denominator)
                    if self.show_progress:
                        progress_loss.update(loss_numerator, loss_denominator)
                except OutOfCUDAMemoryError as e:
                    self.handle_out_of_cuda_memory(
                        dataset,
                        batch,
                        e.info,
                        device,
                        console_logger,
                        event_logger
                    )
                    raise
                batch_size = len(batch)
                if self.show_progress:
                    progress_num_examples += batch_size
                    ticker.progress = batch_no + 1
                    if ticker.tick():
                        progress_loss_value = progress_loss.get_value()
                        progress_duration = datetime.datetime.now() - progress_start_time
                        progress_examples_per_second = progress_num_examples / progress_duration.total_seconds()
                        console_logger.info(
                            f'  {ticker.int_percent}% '
                            f'| loss: {progress_loss_value:.2f} '
                            f'| examples/s: {progress_examples_per_second:.2f}'
                        )
                        progress_loss = LossAccumulator()
                        progress_start_time = datetime.datetime.now()
                        progress_num_examples = 0
                examples_since_checkpoint += batch_size
                if examples_since_checkpoint >= self.examples_per_checkpoint:
                    console_logger.info(f'  checkpoint #{checkpoint_no + 1}')
                    validation_scores = self.evaluate(
                        saver.model,
                        model_interface,
                        dataset,
                        validation_batches
                    )
                    validation_score = validation_scores[validation_metric]
                    console_logger.info(f'    validation cross entropy: {validation_score:.2f}')
                    # Update the learning rate.
                    lr_scheduler.step(validation_score)
                    # Show the current learning rate.
                    curr_learning_rate = optimizer.param_groups[0]['lr']
                    console_logger.info(f'    learning rate: {curr_learning_rate}')
                    # Decide whether to save the model parameters and whether to
                    # stop early.
                    is_best, should_stop = early_stopping.update(validation_score)
                    if is_best:
                        console_logger.info('    saving parameters')
                        saver.save()
                        best_validation_scores = validation_scores
                        best_checkpoint_no = checkpoint_no
                        best_epoch_no = epoch_no
                    event_logger.log('checkpoint', dict(
                        is_best=is_best,
                        scores=validation_scores
                    ))
                    # Reset the count of examples seen since the last checkpoint.
                    # If `examples_since_checkpoint` is not exactly equal to
                    # `self.examples_per_checkpoint` after `batch_size` is
                    # added to it, but is greater than it, include the extra
                    # examples in the updated count.
                    examples_since_checkpoint %= self.examples_per_checkpoint
                    checkpoint_no += 1
                    if should_stop:
                        console_logger.info('  stopping early')
                        break
            if should_stop:
                break
            epoch_loss_value = epoch_loss.get_value()
            epoch_duration = datetime.datetime.now() - epoch_start_time
            console_logger.info(f'  epoch loss: {epoch_loss_value:.2f}')
            console_logger.info(f'  epoch duration: {epoch_duration}')
            if do_profile_memory:
                peak_memory = get_peak_memory(device)
                console_logger.info(f'  peak CUDA memory: {humanfriendly.format_size(peak_memory)}')
            else:
                peak_memory = None
            event_logger.log('epoch', dict(
                loss=epoch_loss_value,
                duration=epoch_duration.total_seconds(),
                peak_memory=peak_memory,
                num_training_batches=len(batches)
            ))
            epoch_no += 1
        total_duration = datetime.datetime.now() - total_start_time
        # TODO Check for this ahead of time.
        if best_validation_scores is None:
            raise ValueError(
                'the maximum number of epochs has been reached, but no '
                'checkpoints have been made'
            )
        best_validation_score = best_validation_scores[validation_metric]
        console_logger.info(f'best validation cross entropy: {best_validation_score:.2f}')
        console_logger.info(f'completed epochs: {epoch_no}')
        console_logger.info(f'best epoch: #{best_epoch_no+1}')
        console_logger.info(f'completed checkpoints: {checkpoint_no}')
        console_logger.info(f'best checkpoint: #{best_checkpoint_no+1}')
        console_logger.info(f'checkpoints since improvement: {early_stopping.updates_since_improvement}')
        console_logger.info(f'total training duration: {total_duration}')
        event_logger.log('train', dict(
            best_validation_scores=best_validation_scores,
            num_epochs=epoch_no,
            best_epoch=best_epoch_no,
            num_checkpoints=checkpoint_no,
            best_checkpoint=best_checkpoint_no,
            checkpoints_since_improvement=early_stopping.updates_since_improvement,
            duration=total_duration.total_seconds()
        ))

    def handle_out_of_cuda_memory(self,
        dataset: Dataset[Example],
        batch: Batch,
        info: dict[str, Any],
        device: torch.device,
        console_logger: logging.Logger,
        event_logger: Logger
    ) -> None:
        console_logger.info('  out of CUDA memory')
        console_logger.info(torch.cuda.memory_summary(device))
        peak_memory = get_peak_memory(device)
        console_logger.info(f'  peak CUDA memory: {humanfriendly.format_size(peak_memory)}')
        logged_info = self.log_failed_batch(dataset, batch, info, console_logger, event_logger)
        event_logger.log('out_of_cuda_memory', dict(
            peak_memory=peak_memory,
            **logged_info
        ))

    def run_parameter_update(self,
        saver: ModelSaver,
        model_interface: ModelInterface,
        dataset: Dataset[Example],
        optimizer: torch.optim.SGD | torch.optim.Adam,
        batch: Batch
    ) -> tuple[float, int]:
        optimizer.zero_grad()
        saver.model.train()
        prepared_batch = None
        try:
            device = model_interface.get_device(None)
            prepared_batch = model_interface.prepare_batch(batch, device, dataset)
            loss_numerators, loss_denominator = self.get_loss(
                saver.model,
                model_interface,
                dataset,
                prepared_batch,
            )
            loss = torch.mean(loss_numerators)
            loss_numerator = torch.sum(loss_numerators.detach()).item()
            del loss_numerators
            loss.backward()
            del loss
            if self.gradient_clipping_threshold is not None:
                torch.nn.utils.clip_grad_norm_(
                    saver.model.parameters(),
                    self.gradient_clipping_threshold
                )
            optimizer.step()
            return loss_numerator, loss_denominator
        except torch.cuda.OutOfMemoryError as e:
            if prepared_batch is not None:
                info = self.get_prepared_batch_info(prepared_batch)
            else:
                info = None
            raise OutOfCUDAMemoryError(info) from e

    def evaluate(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        dataset: Dataset[Example],
        batches: list[Batch]
    ) -> dict[str, Any]:
        model.eval()
        with torch.no_grad():
            cumulative_loss = LossAccumulator()
            for batch in batches:
                device = model_interface.get_device(None)
                prepared_batch = model_interface.prepare_batch(batch, device, dataset)
                numerator, denominator = self.evaluate_batch(
                    model,
                    model_interface,
                    dataset,
                    prepared_batch
                )
                cumulative_loss.update(numerator, denominator)
        return { self.get_validation_metric_name() : cumulative_loss.get_value() }

def get_random_generator_and_seed(random_seed):
    random_seed = get_random_seed(random_seed)
    return random.Random(random_seed), random_seed

def get_random_seed(random_seed):
    return random.getrandbits(32) if random_seed is None else random_seed

class LossAccumulator:

    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0

    def update(self, numerator: float, denominator: int) -> None:
        self.numerator += numerator
        self.denominator += denominator

    def get_value(self) -> float:
        return self.numerator / self.denominator

@dataclasses.dataclass
class OutOfCUDAMemoryError(RuntimeError):
    info: dict[str, Any]
