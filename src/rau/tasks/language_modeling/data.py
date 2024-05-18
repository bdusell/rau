import dataclasses
import pathlib

import torch

from rau.vocab import ToStringVocabulary
from rau.tasks.common.data import load_prepared_data_file
from .vocabulary import load_vocabulary_data_from_file, VocabularyData

@dataclasses.dataclass
class VocabularyContainer:
    input_vocab: ToStringVocabulary
    output_vocab: ToStringVocabulary

def add_data_arguments(parser, validation=True):
    group = parser.add_argument_group('Dataset options')
    group.add_argument('--training-data', type=pathlib.Path,
        help='A directory containing prepared training data. The file '
             '<training-data>/main.prepared will be used as the training '
             'data, and the file <training-data>/main.vocab will be used as '
             'the vocabulary.')
    group.add_argument('--training-data-file', type=pathlib.Path,
        help='A .prepared file containing prepared training data. This '
             'overrides --training-data.')
    if validation:
        group.add_argument('--validation-data', default='validation',
            help='Name of the dataset in the prepared training data directory '
                 'that will be used as validation data. The file '
                 '<training-data>/datasets/<validation-data>/main.prepared '
                 'will be used as the validation data. The default name is '
                 '"validation".')
        group.add_argument('--validation-data-file', type=pathlib.Path,
            help='A .prepared file containing prepared validation data. This '
                 'overrides --validation-data.')
    group.add_argument('--vocabulary-file', type=pathlib.Path,
        help='A .vocab file containing the token vocabulary. This overrides '
             '--training-data.')

def get_training_data_file_path(args, parser):
    if args.training_data_file is not None:
        return args.training_data_file
    elif args.training_data is not None:
        return args.training_data / 'main.prepared'
    else:
        parser.error(
            'either --training-data or --training-data-file is required')

def get_validation_data_file_path(args, parser):
    if args.validation_data_file is not None:
        return args.validation_data_file
    elif args.training_data is not None:
        return args.training_data / 'datasets' / args.validation_data / 'main.prepared'
    else:
        parser.error(
            'either --training-data or --validation-data-file is required')

def get_vocabulary_file_path(args, parser):
    if args.vocabulary_file is not None:
        return args.vocabulary_file
    elif hasattr(args, 'training_data') and args.training_data is not None:
        return args.training_data / 'main.vocab'
    else:
        parser.error(
            'either --training-data or --vocabulary-file is required')

def load_vocabulary_data(args, parser) -> VocabularyData:
    return load_vocabulary_data_from_file(get_vocabulary_file_path(args, parser))

def load_prepared_data(args, parser, vocabulary_data, model_interface, builder=None):
    training_data = load_prepared_data_file(get_training_data_file_path(args, parser))
    if hasattr(args, 'validation_data'):
        validation_data = load_prepared_data_file(get_validation_data_file_path(args, parser))
    else:
        validation_data = None
    input_vocab, output_vocab = model_interface.get_vocabularies(
        vocabulary_data,
        builder
    )
    return (
        training_data,
        validation_data,
        VocabularyContainer(input_vocab, output_vocab)
    )

def load_vocabularies(args, parser, model_interface, builder=None):
    input_vocab, output_vocab = model_interface.get_vocabularies(
        load_vocabulary_data(args, parser),
        builder
    )
    return VocabularyContainer(input_vocab, output_vocab)
