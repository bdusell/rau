import argparse
import random

import torch

from rau.tasks.language_modeling.vocabulary import VocabularyData
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.evaluate import (
    process_sequences_dataset
)

def test_conditional_cross_entropy():
    argv = [
        '--device', 'cpu',
        '--parameter-seed', '123',
        '--architecture', 'transformer',
        '--num-layers', '5',
        '--d-model', '32',
        '--num-heads', '4',
        '--feedforward-size', '64',
        '--dropout', '0',
        '--init-scale', '0.1'
    ]

    model_interface = LanguageModelingModelInterface(use_load=False, use_output=False)
    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    args = parser.parse_args(argv)

    device = model_interface.get_device(args)
    vocabulary_data = VocabularyData(
        tokens=['0', '1'],
        allow_unk=False
    )
    saver = model_interface.construct_saver(args, vocabulary_data)
    model = saver.model

    generator = random.Random(123)
    num_examples = 13
    vocab_size = len(vocabulary_data.tokens)
    def random_string():
        length = generator.randint(0, 10)
        return torch.tensor(
            [generator.randrange(vocab_size) for _ in range(length)],
            dtype=torch.int64,
            device=device
        )
    prompts = [random_string() for _ in range(num_examples)]
    examples = [random_string() for _ in range(num_examples)]
    expected_total_ce = 0.0
    expected_num_tokens = 0
    model.eval()
    with torch.inference_mode():
        for prompt, example in zip(prompts, examples):
            full_example = torch.concat([prompt, example], dim=0)
            full_input_tensor, _ = model_interface.prepare_batch([full_example], device)
            _, example_output_tensor = model_interface.prepare_batch([example], device)
            logits = model_interface.get_logits(model, full_input_tensor)
            example_logits = logits[:, len(prompt):]
            ce = torch.nn.functional.cross_entropy(example_logits.transpose(1, 2), example_output_tensor, reduction='sum')
            expected_total_ce += ce.item()
            expected_num_tokens += len(example) + 1
    expected_ce = expected_total_ce / expected_num_tokens
    result = process_sequences_dataset(
        model,
        model_interface,
        prompts,
        examples,
        max_tokens=10
    )
    torch.testing.assert_close(result, expected_ce)
