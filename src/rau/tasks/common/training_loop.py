import argparse
import collections
import dataclasses
import datetime
import functools
import logging
import random
from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, Optional, TypeVar

import humanfriendly
import torch

from rau.tools.logging import Logger
from rau.tools.ticker import TimedTicker
from rau.tools.torch.saver import ModelSaver
from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.profile import reset_memory_profiler, get_peak_memory
from rau.training.early_stopping import UpdatesWithoutImprovement

Example = TypeVar('Example')
Batch = list[Example]
PreparedBatch = TypeVar('PreparedBatch')
VocabularyContainer = TypeVar('VocabularyContainer')

def add_training_loop_arguments(
    parser: argparse.ArgumentParser,
    max_tokens_per_batch_help: str
):
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
             'on the validation data before training stops early. The minimum '
             'value is 1 (immediate).')
    group.add_argument('--learning-rate-patience', type=int, required=True,
        help='The allowed number of epochs of no improvement in performance '
             'on the validation data before the learning rate is reduced. The '
             'minimum value is 1 (immediate).')
    group.add_argument('--learning-rate-decay-factor', type=float, required=True,
        help='A value between 0 and 1 that the learning rate will be '
             'multiplied by whenever it should be decreased.')
    group.add_argument('--examples-per-checkpoint', type=int, required=True,
        help='An evaluation checkpoint will be run on the validation data '
             'every time this many training examples have been processed.')
    return group

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
class TrainingLoop(Generic[Example, PreparedBatch, VocabularyContainer]):

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
        """Return the name of the validation set metric used for early stopping
        and learning rate scheduling."""
        raise NotImplementedError

    def get_validation_metric_mode(self) -> Literal['min', 'max']:
        """Return whether the validation metric is supposed to go up (max) or
        down (min)."""
        raise NotImplementedError

    def generate_batches(self,
        examples: Iterable[Example],
        max_tokens: int
    ) -> Iterable[Batch]:
        """Given the full list of examples in a dataset and a maximum size,
        group those examples into minibatches."""
        raise NotImplementedError

    def get_prepared_batch_info(self,
        prepared_batch: PreparedBatch
    ) -> dict[str, Any]:
        raise NotImplementedError

    def log_failed_batch(self,
        vocabulary: VocabularyContainer,
        batch: Batch,
        info: Optional[dict[str, Any]],
        console_logger: logging.Logger,
        event_logger: Logger
    ) -> dict[str, Any]:
        raise NotImplementedError

    def get_loss(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        prepared_batch: PreparedBatch
    ) -> (
        tuple[torch.Tensor, float] |
        dict[str, tuple[torch.Tensor, float] | tuple[torch.Tensor, float, float]]
    ):
        """Return a differentiable tensor representing the loss function to be
        optimized."""
        raise NotImplementedError

    def evaluate_batch(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        prepared_batch: PreparedBatch
    ) -> dict[str, tuple[float, float]]:
        raise NotImplementedError

    def run(self,
        saver: ModelSaver,
        model_interface: ModelInterface,
        training_data: list[Example],
        validation_data: list[Example],
        vocabulary: VocabularyContainer,
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
        if self.learning_rate_patience < 1:
            raise ValueError('learning rate patience must be at least 1')
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=validation_metric_mode,
            # According to PyTorch, a patience of 0 means we reduce the LR as
            # soon as performance does not improve, and a patience of 1 means
            # we wait one checkpoint. We subtract 1 so that the patience means
            # the number of epochs without improvement before reducing the LR.
            patience=self.learning_rate_patience - 1,
            factor=self.learning_rate_decay_factor,
            threshold=0.0
        )
        console_logger.info(f'training examples: {len(training_data)}')
        num_validation_examples = len(validation_data)
        console_logger.info(f'validation examples: {num_validation_examples}')
        validation_batches = list(self.generate_batches(
            validation_data,
            self.max_tokens_per_batch
        ))
        console_logger.info(f'validation batches: {len(validation_batches)}')
        model_interface.on_before_process_pairs(
            saver,
            [training_data, validation_data]
        )
        del validation_data
        event_logger.log('start_training', dict(
            num_training_examples=len(training_data),
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
            random_shuffling_generator.shuffle(training_data)
            batches = list(self.generate_batches(
                training_data,
                self.max_tokens_per_batch
            ))
            random_shuffling_generator.shuffle(batches)
            epoch_loss = DictScoreAccumulator()
            if self.show_progress:
                progress_loss = DictScoreAccumulator()
                progress_num_examples = 0
                progress_start_time = datetime.datetime.now()
                ticker = TimedTicker(len(batches), 1)
            if do_profile_memory:
                reset_memory_profiler(device)
            should_stop = False
            for batch_no, batch in enumerate(batches):
                try:
                    loss_numerator, loss_denominator, loss_terms = self.run_parameter_update(
                        saver,
                        model_interface,
                        optimizer,
                        batch
                    )
                    loss_terms['loss'] = (loss_numerator, loss_denominator)
                    epoch_loss.update(loss_terms)
                    if self.show_progress:
                        progress_loss.update(loss_terms)
                except OutOfCUDAMemoryError as e:
                    self.handle_out_of_cuda_memory(
                        vocabulary,
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
                        progress_loss_dict = progress_loss.get_value()
                        progress_loss = progress_loss_dict.pop('loss')
                        progress_duration = datetime.datetime.now() - progress_start_time
                        progress_examples_per_second = progress_num_examples / progress_duration.total_seconds()
                        progress_parts = [
                            f'{ticker.int_percent}%',
                            f'loss: {progress_loss:.2f}',
                            f'examples/s: {progress_examples_per_second:.2f}'
                        ]
                        for key, value in progress_loss_dict.items():
                            progress_parts.append(f'{key}: {value:.2f}')
                        console_logger.info(f'  {" | ".join(progress_parts)}')
                        progress_loss = DictScoreAccumulator()
                        progress_start_time = datetime.datetime.now()
                        progress_num_examples = 0
                examples_since_checkpoint += batch_size
                if examples_since_checkpoint >= self.examples_per_checkpoint:
                    console_logger.info(f'  checkpoint #{checkpoint_no + 1}')
                    validation_scores = self.evaluate(
                        saver.model,
                        model_interface,
                        validation_batches
                    )
                    console_logger.info(f'    validation scores:')
                    for key, value in validation_scores.items():
                        console_logger.info(f'      {key}: {value:.2f}')
                    validation_score = validation_scores[validation_metric]
                    # Update the learning rate.
                    lr_scheduler.step(validation_score)
                    # Show the current learning rate.
                    curr_learning_rate = lr_scheduler.get_last_lr()[0]
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
            epoch_loss_dict = epoch_loss.get_value()
            epoch_loss = epoch_loss_dict.pop('loss')
            epoch_duration = datetime.datetime.now() - epoch_start_time
            epoch_duration_seconds = epoch_duration.total_seconds()
            console_logger.info(f'  epoch loss: {epoch_loss:.2f}')
            if epoch_loss_dict:
                console_logger.info('  epoch scores:')
                for key, value in epoch_loss_dict.items():
                    console_logger.info(f'    {key}: {value:.2f}')
            console_logger.info(f'  epoch duration: {epoch_duration}')
            epoch_examples_per_second = len(training_data) / epoch_duration_seconds
            console_logger.info(f'  examples/s: {epoch_examples_per_second:.2f}')
            if do_profile_memory:
                peak_memory = get_peak_memory(device)
                console_logger.info(f'  peak CUDA memory: {humanfriendly.format_size(peak_memory)}')
            else:
                peak_memory = None
            event_logger.log('epoch', dict(
                loss=epoch_loss,
                scores=epoch_loss_dict,
                duration=epoch_duration_seconds,
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
        console_logger.info(f'best validation scores:')
        for key, value in best_validation_scores.items():
            console_logger.info(f'  {key}: {value:.2f}')
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
        vocabulary: VocabularyContainer,
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
        logged_info = self.log_failed_batch(vocabulary, batch, info, console_logger, event_logger)
        event_logger.log('out_of_cuda_memory', dict(
            peak_memory=peak_memory,
            **logged_info
        ))

    def run_parameter_update(self,
        saver: ModelSaver,
        model_interface: ModelInterface,
        optimizer: torch.optim.SGD | torch.optim.Adam,
        batch: Batch
    ) -> tuple[
        float,
        float,
        dict[str, tuple[float, float]]
    ]:
        # Reset the parameters' gradients to 0.
        optimizer.zero_grad()
        # Activate training mode (activate dropout, etc.).
        saver.model.train()
        # Tensorize the batch, run the forward pass, and get the loss.
        (
            prepared_batch,
            loss,
            loss_numerator,
            loss_denominator,
            extra_loss_terms
        ) = self.get_prepared_batch_and_loss(
            saver,
            model_interface,
            batch
        )
        try:
            # Run backprop to compute the gradients.
            loss.backward()
            # Free the computation graph from memory.
            del loss
            # Do gradient clipping.
            if self.gradient_clipping_threshold is not None:
                torch.nn.utils.clip_grad_norm_(
                    saver.model.parameters(),
                    self.gradient_clipping_threshold
                )
            # Update the parameters using the gradients.
            optimizer.step()
        except torch.cuda.OutOfMemoryError as e:
            info = self.get_prepared_batch_info(prepared_batch)
            raise OutOfCUDAMemoryError(info) from e
        return loss_numerator, loss_denominator, extra_loss_terms

    def get_prepared_batch_and_loss(self,
        saver: ModelSaver,
        model_interface: ModelInterface,
        batch: Batch
    ) -> tuple[
        PreparedBatch,
        torch.Tensor,
        float,
        float,
        dict[str, tuple[float, float]]
    ]:
        prepared_batch = None
        try:
            # Tensorize the minibatch.
            device = model_interface.get_device(None)
            prepared_batch = model_interface.prepare_batch(batch, device)
            # Run the forward pass and get the loss (or multiple loss terms).
            loss_result = self.get_loss(
                saver.model,
                model_interface,
                prepared_batch
            )
            extra_loss_terms = {}
            if isinstance(loss_result, dict):
                # There are multiple loss terms. Add them together, using
                # coefficients if given.
                loss_terms = []
                loss_numerator = 0
                loss_denominator = 0
                for key, value in loss_result.items():
                    try:
                        loss_term_numerators, loss_term_denominator, coefficient = value
                    except ValueError:
                        loss_term_numerators, loss_term_denominator = value
                        coefficient = None
                    # Sum up all of the numerators. We will divide all of the
                    # numereators by the number of examples in the batch at the
                    # end to get the average. Not all loss terms necessarily
                    # have a value for every example.
                    loss_term_sum = torch.sum(loss_term_numerators)
                    # Return the unweighted numerator/denominator for each loss
                    # term.
                    loss_term_numerator = loss_term_sum.item()
                    extra_loss_terms[key] = (loss_term_numerator, loss_term_denominator)
                    # If this loss term has a coefficient, multiply it into the
                    # loss term, its numerator, and its denominator.
                    if coefficient is not None:
                        loss_term_sum = loss_term_sum * coefficient
                        loss_term_numerator *= coefficient
                        loss_term_denominator *= coefficient
                    loss_terms.append(loss_term_sum)
                    loss_numerator += loss_term_numerator
                    loss_denominator += loss_term_denominator
                # Divide the sum of all the numerators by the number of examples
                # in the batch in order to get the mean loss. Not all loss
                # terms necessarily have a value for every example.
                loss = functools.reduce(lambda x, y: x + y, loss_terms) / len(batch)
                del loss_term_numerators
            else:
                # There is only one loss term.
                loss_numerators, loss_denominator = loss_result
                # Get the mean loss.
                loss = torch.mean(loss_numerators)
                loss_numerator = torch.sum(loss_numerators.detach()).item()
                del loss_numerators
            del loss_result
        except torch.cuda.OutOfMemoryError as e:
            if prepared_batch is not None:
                info = self.get_prepared_batch_info(prepared_batch)
            else:
                info = None
            raise OutOfCUDAMemoryError(info) from e
        return (
            prepared_batch,
            loss,
            loss_numerator,
            loss_denominator,
            extra_loss_terms
        )

    def evaluate(self,
        model: torch.nn.Module,
        model_interface: ModelInterface,
        batches: list[Batch]
    ) -> dict[str, float]:
        return evaluate(
            model,
            model_interface,
            batches,
            self.evaluate_batch
        )

def get_random_generator_and_seed(random_seed):
    random_seed = get_random_seed(random_seed)
    return random.Random(random_seed), random_seed

def get_random_seed(random_seed):
    return random.getrandbits(32) if random_seed is None else random_seed

class MicroAveragedScoreAccumulator:

    def __init__(self):
        super().__init__()
        self.numerator = 0
        self.denominator = 0

    def update(self, numerator: float, denominator: float) -> None:
        self.numerator += numerator
        self.denominator += denominator

    def get_value(self) -> float:
        return self.numerator / self.denominator

class DictScoreAccumulator:

    def __init__(self):
        super().__init__()
        self.loss = None

    def update(self, scores: dict[str, tuple[float, float]]) -> None:
        if self.loss is None:
            self.loss = { k : MicroAveragedScoreAccumulator() for k in scores.keys() }
        elif scores.keys() != self.loss.keys():
            raise ValueError
        for key, (numerator, denominator) in scores.items():
            self.loss[key].update(numerator, denominator)

    def get_value(self) -> dict[str, float]:
        return { k : v.get_value() for k, v in self.loss.items() }

def evaluate(
    model: torch.nn.Module,
    model_interface: ModelInterface,
    batches: list[Batch],
    evaluate_batch: Callable[
        [torch.nn.Module, ModelInterface, PreparedBatch],
        dict[str, tuple[float, float]]
    ]
) -> dict[str, float]:
    device = model_interface.get_device(None)
    accumulator = DictScoreAccumulator()
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            prepared_batch = model_interface.prepare_batch(batch, device)
            batch_score_dict = evaluate_batch(
                model,
                model_interface,
                prepared_batch
            )
            accumulator.update(batch_score_dict)
    return accumulator.get_value()

@dataclasses.dataclass
class OutOfCUDAMemoryError(RuntimeError):
    info: dict[str, Any]
