import argparse

from rau.tasks.language_modeling.model_size import (
    LanguageModelingModelSizeCommand,
    get_arg_dict,
    get_transformer_num_parameters
)
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import VocabularyData

def get_actual_num_parameters(argv, vocabulary_data):
    model_interface = LanguageModelingModelInterface()
    parser = argparse.ArgumentParser()
    model_interface.add_more_init_arguments(parser)
    args = parser.parse_args(argv)
    kwargs = model_interface.get_kwargs(args, vocabulary_data)
    model = model_interface.construct_model(**kwargs)
    for name, param in model.named_parameters():
        print(name, param.numel())
    num_params = sum(p.numel() for p in model.parameters())
    return num_params, kwargs['input_vocabulary_size']

def get_vocabulary_data(vocab_size):
    return VocabularyData(
        tokens=[str(i) for i in range(vocab_size)],
        allow_unk=False
    )

def test_transformer_num_parameters():
    vocab_size = 13
    num_layers = 5
    d_model = 21
    num_heads = 7
    feedforward_size = 17
    vocabulary_data = get_vocabulary_data(vocab_size)
    expected_num_params, num_embeddings = get_actual_num_parameters([
        '--architecture', 'transformer',
        '--num-layers', str(num_layers),
        '--d-model', str(d_model),
        '--num-heads', str(num_heads),
        '--feedforward-size', str(feedforward_size),
        '--dropout', '0.1'
    ], vocabulary_data)
    num_params = get_transformer_num_parameters(
        num_embeddings=num_embeddings,
        d_model=d_model,
        num_layers=num_layers,
        feedforward_size=feedforward_size
    )
    assert num_params == expected_num_params

def run_resize(argv, vocabulary_data):
    command = LanguageModelingModelSizeCommand()
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    args = parser.parse_args(argv)
    return get_arg_dict(args, vocabulary_data)

def test_transformer_resize():
    target_num_params = 123000
    vocab_size = 13
    num_layers = 5
    d_model = 21
    num_heads = 7
    feedforward_size_factor = 3
    vocabulary_data = get_vocabulary_data(vocab_size)
    arg_dict = run_resize([
        '--parameters', str(target_num_params),
        '--architecture', 'transformer',
        '--num-layers', str(num_layers),
        '--num-heads', str(num_heads),
        '--feedforward-size-factor', str(feedforward_size_factor)
    ], vocabulary_data)
    d_model = arg_dict['--d-model']

    def get_num_params(d_model):
        updated_arg_dict = arg_dict | { '--d-model' : d_model, '--dropout' : 0.1 }
        num_params, _ = get_actual_num_parameters(
            [str(arg) for pair in updated_arg_dict.items() for arg in pair],
            vocabulary_data
        )
        return num_params

    assert_is_closest(
        target_num_params,
        get_num_params(d_model),
        get_num_params(d_model - num_heads),
        get_num_params(d_model + num_heads)
    )

def assert_is_closest(target, result, lo, hi):
    assert abs(target - result) < abs(target - lo)
    assert abs(target - result) < abs(target - hi)
