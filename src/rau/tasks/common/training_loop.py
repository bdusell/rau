import dataclasses
import random

import torch

def add_training_loop_arguments(
    parser,
    batching_max_tokens_help,
    use_label_smoothing
):
    group = parser.add_argument_group('Training options')
    group.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages during training.')
    group.add_argument('--epochs', type=int, required=True,
        help='The maximum number of epochs to run training for.')
    group.add_argument('--random-shuffling-seed', type=int,
        help='Random seed used for random shuffling of the training data.')
    group.add_argument('--batching-max-tokens', type=int, required=True,
        help=batching_max_tokens_help)
    add_optimizer_arguments(group)
    add_parameter_update_arguments(group, use_label_smoothing)
    group.add_argument('--early-stopping-patience', type=int, required=True,
        help='The allowed number of epochs of no improvement in performance '
             'on the validation data before training stops early.')
    group.add_argument('--learning-rate-patience', type=int, required=True,
        help='The allowed number of epochs of no improvement in performance '
             'on the validation data before the learning rate is reduced.')
    group.add_argument('--learning-rate-decay-factor', type=float, required=True,
        help='A value between 0 and 1 that the learning rate will be '
             'multiplied by whenever it should be decreased.')
    group.add_argument('--checkpoint-interval-sequences', type=int, required=True,
        help='An evaluation checkpoint will be run on the validation data '
             'every time this many training examples have been processed.')

def add_optimizer_arguments(group):
    group.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam',
        help='The algorithm to use for parameter optimization.')
    group.add_argument('--learning-rate', type=float, required=True,
        help='The initial learning rate.')

def add_parameter_update_arguments(group, use_label_smoothing):
    if use_label_smoothing:
        group.add_argument('--label-smoothing', type=float, default=0.0,
            help='The label smoothing factor to use with the cross-entropy '
                 'loss. Default is 0 (no label smoothing).')
    group.add_argument('--gradient-clipping-threshold', type=float,
        help='The threshold to use for L2 gradient clipping. If not given, '
             'gradients are never clipped.')

def get_random_generator_and_seed(random_seed):
    random_seed = get_random_seed(random_seed)
    return random.Random(random_seed), random_seed

def get_random_seed(random_seed):
    return random.getrandbits(32) if random_seed is None else random_seed

def get_optimizer(saver, args):
    OptimizerClass = getattr(torch.optim, args.optimizer)
    return OptimizerClass(
        saver.model.parameters(),
        lr=args.learning_rate
    )

@dataclasses.dataclass
class ParameterUpdateResult:
    loss_numer: float
    num_symbols: int

class LossAccumulator:

    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0

    def update(self, numerator, denominator):
        self.numerator += numerator
        self.denominator += denominator

    def get_value(self):
        return self.numerator / self.denominator
