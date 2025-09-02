import argparse

from rau.tasks.language_modeling.model_size import (
    LanguageModelingModelSizeCommand,
    get_arg_dict,
    get_transformer_num_parameters
)
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import VocabularyData

def test_transformer_num_parameters():
    model_interface = LanguageModelingModelInterface()
    vocab_size = 13
    num_layers = 5
    d_model = 21
    num_heads = 7
    feedforward_size = 17
    vocabulary_data = VocabularyData(
        tokens=[str(i) for i in range(vocab_size)],
        allow_unk=False
    )
    parser = argparse.ArgumentParser()
    model_interface.add_more_init_arguments(parser)
    argv = [
        '--architecture', 'transformer',
        '--num-layers', str(num_layers),
        '--d-model', str(d_model),
        '--num-heads', str(num_heads),
        '--feedforward-size', str(feedforward_size),
        '--dropout', '0.1'
    ]
    args = parser.parse_args(argv)
    kwargs = model_interface.get_kwargs(args, vocabulary_data)
    model = model_interface.construct_model(**kwargs)
    expected_num_params = sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        print(name, param.numel())
    num_params = get_transformer_num_parameters(
        num_embeddings=kwargs['input_vocabulary_size'],
        d_model=d_model,
        num_layers=num_layers,
        feedforward_size=feedforward_size
    )
    assert num_params == expected_num_params

def test_transformer_resize():
    target_num_params = 123000
    vocab_size = 13
    num_layers = 5
    d_model = 21
    num_heads = 7
    feedforward_size_factor = 3
    vocabulary_data = VocabularyData(
        tokens=[str(i) for i in range(vocab_size)],
        allow_unk=False
    )
    command = LanguageModelingModelSizeCommand()
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    argv = [
        '--parameters', str(target_num_params),
        '--architecture', 'transformer',
        '--num-layers', str(num_layers),
        '--num-heads', str(num_heads),
        '--feedforward-size-factor', str(feedforward_size_factor)
    ]
    args = parser.parse_args(argv)
    arg_dict = get_arg_dict(args, vocabulary_data)
    d_model = arg_dict['--d-model']

    def get_num_params(d_model):
        updated_arg_dict = arg_dict | { '--d-model' : d_model, '--dropout' : 0.1 }
        model_interface = LanguageModelingModelInterface()
        parser = argparse.ArgumentParser()
        model_interface.add_more_init_arguments(parser)
        argv = [str(arg) for pair in updated_arg_dict.items() for arg in pair]
        args = parser.parse_args(argv)
        kwargs = model_interface.get_kwargs(args, vocabulary_data)
        model = model_interface.construct_model(**kwargs)
        return sum(p.numel() for p in model.parameters())

    num_params = get_num_params(d_model)
    num_params_lo = get_num_params(d_model - num_heads)
    num_params_hi = get_num_params(d_model + num_heads)
    assert abs(target_num_params - num_params) < abs(target_num_params - num_params_lo)
    assert abs(target_num_params - num_params) < abs(target_num_params - num_params_hi)
