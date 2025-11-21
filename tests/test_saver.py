import torch

from rau.tools.torch.saver import ModelSaver

def test_to_directory(tmp_path):

    def construct_model(input_size, output_size):
        return torch.nn.Linear(input_size, output_size)

    original_output = tmp_path / 'original'
    new_output = tmp_path / 'new'
    saver = ModelSaver.construct(construct_model, original_output, input_size=7, output_size=13)
    saver.save_kwargs()
    saver.save_parameters()
    saver = ModelSaver.read(construct_model, original_output)
    saver = saver.to_directory(new_output)
    saver.save_kwargs()
    saver.save_parameters()
    saver = ModelSaver.read(construct_model, new_output)
