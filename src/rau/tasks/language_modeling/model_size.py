import argparse
import pathlib
from typing import Any

import sympy

from rau.tasks.common.command import Command
from rau.tasks.language_modeling.data import load_vocabulary_data, VocabularyData
from rau.models.stack_nn.transformer.parse import (
    parse_stack_transformer_layers,
    STACK_TRANSFORMER_LAYERS_HELP_MESSAGE
)
from rau.models.stack_nn.rnn.parse import (
    parse_stack_rnn_stack
)

class LanguageModelingModelSizeCommand(Command):

    DESCRIPTION = (
        'Given a parameter count and some architecture hyperparameters, print '
        'command-line arguments for the train command that will result in a '
        'model with that parameter count, or as close to it as possible.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--training-data', type=pathlib.Path,
            help='A directory containing prepared training data. The file '
                '<training-data>/main.vocab will be used as the vocabulary.')
        parser.add_argument('--vocabulary-file', type=pathlib.Path,
            help='A .vocab file containing the token vocabulary. This '
                 'overrides --training-data.')
        parser.add_argument('--parameters', type=int, required=True,
            help='The target parameter count.')
        parser.add_argument('--architecture',
            choices=['transformer', 'rnn', 'lstm', 'stack-transformer', 'stack-rnn'],
            help='The type of neural network architecture to use. '
                 'transformer: Standard transformer architecture with '
                 'sinusoidal positional encodings, based on the PyTorch '
                 'implementation. '
                 'rnn: Simple RNN (a.k.a. Elman RNN) with tanh activations. '
                 'lstm: LSTM. '
                 'stack-transformer: A transformer that can contain stack '
                 'attention layers. '
                 'stack-rnn: An RNN controller augmented with a '
                 'differentiable stack.')
        parser.add_argument('--num-layers', type=int,
            help='(transformer, rnn, lstm) Number of layers.')
        parser.add_argument('--num-heads', type=int,
            help='(transformer) The number of attention heads used in each '
                 'layer.')
        parser.add_argument('--feedforward-size-factor', type=int,
            help='(transformer) The size of the hidden layer of the '
                 'feedforward network in each feedforward sublayer.')
        parser.add_argument('--stack-transformer-layers', type=parse_stack_transformer_layers,
            help='(stack-transformer) A string describing which layers to '
                 'use. ' +
                 STACK_TRANSFORMER_LAYERS_HELP_MESSAGE)
        parser.add_argument('--stack-rnn-controller', choices=['rnn', 'lstm'],
            help='(stack-rnn) The type of RNN to use as the controller.')
        parser.add_argument('--stack-rnn-stack', type=parse_stack_rnn_stack,
            help='(stack-rnn) The type of differentiable stack to connect to '
                 'the RNN controller. Options are: '
                 '(1) stratification-<m>, where <m> is an integer, indicating '
                 'a stratification stack with stack embedding size <m> '
                 '(2) superposition-<m>, where <m> is an integer, indicating '
                 'a superposition stack with stack embedding size <m> '
                 '(3) nondeterministic-<q>-<s>, where <q>, <s> are integers, '
                 'indicating a nondeterministic stack with <q> states and <s> '
                 'stack symbol types '
                 '(4) vector-nondeterministic-<q>-<s>-<m>, where <q>, <s>, '
                 '<m> are integers, indicating a vector nondeterministic '
                 'stack with <q> states, <s> stack symbol types, and stack '
                 'embedding type <m>.')
        parser.add_argument('--stack-rnn-connect-reading-to-output', action='store_true', default=False,
            help='(stack-rnn) Connect the stack reading directly to the output '
                 'at the same timestep when making predictions.')

    def run(self, parser, args):
        arg_dict = get_arg_dict(args, load_vocabulary_data(args, parser))
        print(' '.join(str(arg) for pair in arg_dict.items() for arg in pair))

def get_arg_dict(
    args: argparse.Namespace,
    vocabulary_data: VocabularyData
) -> dict[str, Any]:
    match args.architecture:
        case 'transformer' | 'stack-transformer':
            if args.num_layers is None:
                raise ValueError
            if args.num_heads is None:
                raise ValueError
            if args.feedforward_size_factor is None:
                raise ValueError
            size = sympy.Symbol('size', positive=True)
            d_model = size * args.num_heads
            feedforward_size = args.feedforward_size_factor * d_model
            num_embeddings = len(vocabulary_data.tokens) + int(vocabulary_data.allow_unk) + 2
            match args.architecture:
                case 'transformer':
                    num_params_expr = get_transformer_num_parameters(
                        num_embeddings=num_embeddings,
                        d_model=d_model,
                        num_layers=args.num_layers,
                        feedforward_size=feedforward_size
                    )
                case 'stack-transformer':
                    if args.stack_transformer_layers is None:
                        raise ValueError
                    num_params_expr = get_stack_transformer_num_parameters(
                        num_embeddings=num_embeddings,
                        d_model=d_model,
                        feedforward_size=feedforward_size,
                        stack_transformer_layers=args.stack_transformer_layers
                    )
                case _:
                    raise ValueError
            eq = sympy.Eq(num_params_expr, args.parameters)
            size_expr = sympy.solve(eq, size, dict=True)[0][size]
            size_int = round(size_expr.evalf())
            d_model_int = int(d_model.evalf(subs={ size : size_int }))
            feedforward_size_int = int(feedforward_size.evalf(subs={ size : size_int }))
            return {
                '--architecture' : args.architecture,
                '--num-layers' : args.num_layers,
                '--d-model' : d_model_int,
                '--num-heads' : args.num_heads,
                '--feedforward-size' : feedforward_size_int
            }
        case 'rnn' | 'lstm':
            if args.num_layers is None:
                raise ValueError
            hidden_units = sympy.Symbol('size', positive=True)
            eq = sympy.Eq(
                get_rnn_num_parameters(
                    architecture=args.architecture,
                    num_embeddings=len(vocabulary_data.tokens) + int(vocabulary_data.allow_unk) + 1,
                    num_layers=args.num_layers,
                    hidden_units=hidden_units
                ),
                args.parameters
            )
            hidden_units_expr = sympy.solve(eq, hidden_units, dict=True)[0][hidden_units]
            hidden_units_int = round(hidden_units_expr.evalf())
            return {
                '--architecture' : args.architecture,
                '--num-layers' : args.num_layers,
                '--hidden-units' : hidden_units_int
            }

def get_transformer_num_parameters(
    num_embeddings,
    d_model,
    num_layers,
    feedforward_size
):
    return (
        num_embeddings * d_model + # embeddings
        num_layers * get_transformer_layer_num_parameters(d_model, feedforward_size) +
        2 * d_model # final layer norm
    )

def get_transformer_layer_num_parameters(d_model, feedforward_size):
    return (
        3 * (d_model + 1) * d_model + # in projection layers
        (d_model + 1) * d_model + # out projection layers
        (feedforward_size + 1) * d_model + # feedforward first layer
        (d_model + 1) * feedforward_size + # feedforward second layer
        4 * d_model # layer norm
    )

def get_stack_transformer_num_parameters(
    num_embeddings,
    d_model,
    feedforward_size,
    stack_transformer_layers
):
    return (
        num_embeddings * d_model + # embeddings
        sum(
            get_stack_transformer_layer_num_parameters(layer, d_model, feedforward_size)
            for layer in stack_transformer_layers
        ) +
        2 * d_model # final layer norm
    )

def get_stack_transformer_layer_num_parameters(layer, d_model, feedforward_size):
    match layer:
        case ('transformer', (num_layers,)):
            return num_layers * get_transformer_layer_num_parameters(d_model, feedforward_size)
        case ('superposition', (stack_embedding_size,)):
            num_stack_actions = 3
            return (
                4 * d_model + # two layer norms
                num_stack_actions * d_model + # stack action layer
                stack_embedding_size * d_model + # input to pushed vector layer
                d_model * stack_embedding_size + # stack reading to output layer
                (feedforward_size + 1) * d_model + # feedforward first layer
                (d_model + 1) * feedforward_size # feedforward second layer
            )
        case _:
            raise ValueError

def get_rnn_num_parameters(
    architecture,
    num_embeddings,
    num_layers,
    hidden_units
):
    match architecture:
        case 'rnn':
            num_gates = 1
        case 'lstm':
            num_gates = 4
        case _:
            raise ValueError
    return (
        num_embeddings * hidden_units + # embeddings
        num_layers * hidden_units + # initial hidden state
        num_layers * (
            num_gates * (2 * hidden_units + 1) * hidden_units # input/recurrent layers
        )
    )

if __name__ == '__main__':
    LanguageModelingModelSizeCommand().main()
