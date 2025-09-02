import argparse

from rau.tasks.language_modeling.model_size import (
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
