import argparse
import logging

from rau.tasks.common.command import get_logger
from rau.tasks.language_modeling.prepare_data import LanguageModelingPrepareDataCommand
from rau.tasks.language_modeling.model_size import LanguageModelingModelSizeCommand
from rau.tasks.language_modeling.train import LanguageModelingTrainCommand
from rau.tasks.language_modeling.evaluate import LanguageModelingEvaluateCommand
from rau.tasks.language_modeling.generate import LanguageModelingGenerateCommand
from rau.tasks.sequence_to_sequence.prepare_data import SequenceToSequencePrepareDataCommand
from rau.tasks.sequence_to_sequence.train import SequenceToSequenceTrainCommand
from rau.tasks.sequence_to_sequence.translate import SequenceToSequenceTranslateCommand
from rau.tasks.common.is_finished import IsFinishedCommand

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands for tasks', dest='task_command', required=True)

    console_logger = get_logger()
    lm_prepare_command = LanguageModelingPrepareDataCommand()
    lm_model_size_command = LanguageModelingModelSizeCommand()
    lm_train_command = LanguageModelingTrainCommand(console_logger)
    lm_evaluate_command = LanguageModelingEvaluateCommand()
    lm_generate_command = LanguageModelingGenerateCommand()
    ss_prepare_command = SequenceToSequencePrepareDataCommand()
    ss_train_command = SequenceToSequenceTrainCommand(console_logger)
    ss_translate_command = SequenceToSequenceTranslateCommand(console_logger)
    is_finished_command = IsFinishedCommand()

    # Language Modeling
    lm_parser = subparsers.add_parser('lm', help='Language modeling.')
    lm_subparsers = lm_parser.add_subparsers(dest='lm_command', required=True)
    lm_prepare_command.add_arguments(lm_subparsers.add_parser('prepare',
        help='Prepare data for use by a neural language model.',
        description=lm_prepare_command.description()
    ))
    lm_model_size_command.add_arguments(lm_subparsers.add_parser('model-size',
        help='Get hyperparameters corresponding to a particular parameter count.',
        description=lm_model_size_command.description()
    ))
    lm_train_command.add_arguments(lm_subparsers.add_parser('train',
        help='Train a neural language model.',
        description=lm_train_command.description()
    ))
    lm_evaluate_command.add_arguments(lm_subparsers.add_parser('evaluate',
        help='Evaluate a neural language model.',
        description=lm_evaluate_command.description()
    ))
    lm_generate_command.add_arguments(lm_subparsers.add_parser('generate',
        help='Sample strings from a neural language model.',
        description=lm_generate_command.description()
    ))

    # Sequence-to-Sequence
    ss_parser = subparsers.add_parser('ss', help='Sequence-to-sequence transduction.')
    ss_subparsers = ss_parser.add_subparsers(dest='ss_command', required=True)
    ss_prepare_command.add_arguments(ss_subparsers.add_parser('prepare',
        help='Prepare data for use by a neural sequence-to-sequence model.',
        description=ss_prepare_command.description()
    ))
    ss_train_command.add_arguments(ss_subparsers.add_parser('train',
        help='Train a neural sequence-to-sequence model.',
        description=ss_train_command.description()
    ))
    ss_translate_command.add_arguments(ss_subparsers.add_parser('translate',
        help='Translate input sequences to output sequences using a neural sequence-to-sequence model.',
        description=ss_translate_command.description()
    ))

    is_finished_command.add_arguments(subparsers.add_parser('is-finished',
        help='Tell whether training for a saved model is finished.',
        description=is_finished_command.description()
    ))

    args = parser.parse_args()

    match args.task_command:
        case 'lm':
            match args.lm_command:
                case 'prepare':
                    lm_prepare_command.run(parser, args)
                case 'model-size':
                    lm_model_size_command.run(parser, args)
                case 'train':
                    lm_train_command.run(parser, args)
                case 'evaluate':
                    lm_evaluate_command.run(parser, args)
                case 'generate':
                    lm_generate_command.run(parser, args)
        case 'ss':
            match args.ss_command:
                case 'prepare':
                    ss_prepare_command.run(parser, args)
                case 'train':
                    ss_train_command.run(parser, args)
                case 'translate':
                    ss_translate_command.run(parser, args)
        case 'is-finished':
            is_finished_command.run(parser, args)

if __name__ == '__main__':
    main()
