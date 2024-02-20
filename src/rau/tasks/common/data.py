import torch

def load_prepared_data_file(path):
    return [torch.tensor(x) for x in torch.load(path)]
