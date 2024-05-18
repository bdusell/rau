import dataclasses
import pathlib

from rau.vocab import ToStringVocabulary, ToStringVocabularyBuilder
from rau.tasks.common.data import load_prepared_data_file
from .vocabulary import (
    load_shared_vocabulary_data_from_file,
    SharedVocabularyData
)

@dataclasses.dataclass
class VocabularyContainer:
    source_vocab: ToStringVocabulary
    target_input_vocab: ToStringVocabulary
    target_output_vocab: ToStringVocabulary
    is_shared: bool

def add_data_arguments(parser, validation=True):
    group = parser.add_argument_group('Dataset options')
    group.add_argument('--training-data', type=pathlib.Path,
        help='A directory containing prepared training data. The file '
             '<training-data>/source.<vocabulary-type>.prepared will be used '
             'as the source side of the training data, and the file '
             '<training-data>/target.<vocabulary-type>.prepared will be used '
             'as the target side. The vocabulary will be read from this '
             'directory as well.')
    group.add_argument('--training-data-source-file', type=pathlib.Path,
        help='A .prepared file containing prepared source sequences of '
             'training data. This overrides --training-data.')
    group.add_argument('--training-data-target-file', type=pathlib.Path,
        help='A .prepared file containing prepared target sequences of '
             'training data. This overrides --training-data.')
    if validation:
        group.add_argument('--validation-data', default='validation',
            help='Name of the dataset in the prepared training data directory '
                 'that will be used as validation data. The files in '
                 '<training-data>/datasets/<validation-data> will be used as '
                 'the validation data. The default name is "validation".')
        group.add_argument('--validation-data-source-file', type=pathlib.Path,
            help='A .prepared file containing prepared source sequences of '
                 'validation data. This overrides --validation-data.')
        group.add_argument('--validation-data-target-file', type=pathlib.Path,
            help='A .prepared file containing prepared target sequences of '
                 'validation data. This overrides --validation-data.')
    group.add_argument('--vocabulary-type', choices=['shared', 'separate'],
        help='Whether to use a single shared vocabulary or two separate '
             'vocabularies for the source and target sides. If "shared" is '
             'used, then the file '
             '<training-data>/shared.vocab will be used as the vocabulary. '
             'If "separate" is used, then the file '
             '<training-data>/source.vocab will be used as the source '
             'vocabulary, and the file '
             '<training-data>/target.vocab will be used as the target '
             'vocabulary.')
    group.add_argument('--shared-vocabulary-file', type=pathlib.Path,
        help='A .vocab file to be used as a shared source-target vocabulary. '
             'This overrides --training-data.')
    group.add_argument('--source-vocabulary-file', type=pathlib.Path,
        help='A .vocab file to be used as the source vocabulary. This '
             'overrides --training-data.')
    group.add_argument('--target-vocabulary-file', type=pathlib.Path,
        help='A .vocab file to be used as the target vocabulary. This '
             'overrides --training-data.')

def get_training_data_source_file_path(args, parser):
    if args.training_data_source_file is not None:
        return args.training_data_source_file
    elif args.training_data is not None:
        return args.training_data / f'source.{args.vocabulary_type}.prepared'
    else:
        parser.error(
            'either --training-data or --training-data-source-file is required')

def get_training_data_target_file_path(args, parser):
    if args.training_data_target_file is not None:
        return args.training_data_target_file
    elif args.training_data is not None:
        return args.training_data / f'target.{args.vocabulary_type}.prepared'
    else:
        parser.error(
            'either --training-data or --training-data-target-file is required')

def get_validation_data_source_file_path(args, parser):
    if args.validation_data_source_file is not None:
        return args.validation_data_source_file
    elif args.training_data is not None:
        return args.training_data / 'datasets' / args.validation_data / f'source.{args.vocabulary_type}.prepared'
    else:
        parser.error(
            'either --training-data or --validation-data-source-file is required')

def get_validation_data_target_file_path(args, parser):
    if args.validation_data_target_file is not None:
        return args.validation_data_target_file
    elif args.training_data is not None:
        return args.training_data / 'datasets' / args.validation_data / f'target.{args.vocabulary_type}.prepared'
    else:
        parser.error(
            'either --training-data or --validation-data-target-file is required')

def get_vocabulary_file_paths(args, parser):
    if args.shared_vocabulary_file is not None:
        if args.source_vocabulary_file is not None or args.target_vocabulary_file is not None:
            parser.error(
                'cannot pass both --shared-vocabulary-file and '
                '--source-vocabulary-file or --target-vocabulary-file')
        else:
            return (args.shared_vocabulary_file,)
    elif args.source_vocabulary_file is not None and args.target_vocabulary_file is not None:
        return (args.source_vocabulary_file, args.target_vocabulary_file)
    elif (
        hasattr(args, 'vocabulary_type') and
        hasattr(args, 'training_data') and
        args.vocabulary_type is not None
    ):
        if args.vocabulary_type == 'shared':
            return (args.training_data / 'shared.vocab',)
        else:
            return (args.training_data / 'source.vocab', args.training_data / 'target.vocab')
    else:
        parser.error(
            'one of --training-data and --vocabulary-type, '
            '--shared-vocabulary-file, or --source-vocabulary-file and '
            '--target-vocabulary-file is required')

    paths = get_vocabulary_file_paths(args, parser)
    return load_vocabulary_data_from_file(get_vocabulary_file_paths(args, parser))

def load_vocabulary_data(args, parser) -> SharedVocabularyData:
    file_paths = get_vocabulary_file_paths(args, parser)
    if len(file_paths) == 1:
        file_path, = file_paths
        return load_shared_vocabulary_data_from_file(file_path)
    else:
        raise NotImplementedError(
            'using separate source and target vocabularies is not yet implemented')

def load_prepared_data(args, parser, vocabulary_data, model_interface, builder=None):
    training_data = load_prepared_data_files(
        get_training_data_source_file_path(args, parser),
        get_training_data_target_file_path(args, parser)
    )
    if hasattr(args, 'validation_data'):
        validation_data = load_prepared_data_files(
            get_validation_data_source_file_path(args, parser),
            get_validation_data_target_file_path(args, parser)
        )
    else:
        validation_data = None
    source_vocab, target_input_vocab, target_output_vocab = model_interface.get_vocabularies(
        vocabulary_data,
        builder
    )
    return (
        training_data,
        validation_data,
        VocabularyContainer(
            source_vocab=source_vocab,
            target_input_vocab=target_input_vocab,
            target_output_vocab=target_output_vocab,
            is_shared=True
        )
    )

def load_prepared_data_files(source_path, target_path):
    return list(zip(
        load_prepared_data_file(source_path),
        load_prepared_data_file(target_path),
        strict=True
    ))

def load_vocabularies(args, parser, model_interface, builder=None):
    (
        source_vocab,
        target_input_vocab,
        target_output_vocab
    ) = model_interface.get_vocabularies(
        load_vocabulary_data(args, parser),
        builder
    )
    return VocabularyContainer(
        source_vocab,
        target_input_vocab,
        target_output_vocab,
        is_shared=True
    )
