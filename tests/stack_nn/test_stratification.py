import torch

from rau.models.rnn.lstm import LSTM
from rau.models.stack_nn.rnn.stratification import StratificationStackRNN

def test_stratification():
    stack_embedding_size = 3
    input_size = 5
    hidden_units = 7
    batch_size = 11
    sequence_length = 13
    generator = torch.manual_seed(0)
    def controller(input_size):
        return LSTM(input_size, hidden_units)
    model = StratificationStackRNN(
        input_size=input_size,
        stack_embedding_size=stack_embedding_size,
        controller=controller,
        controller_output_size=hidden_units,
        include_reading_in_output=False
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
    input_tensor.uniform_(generator=generator)
    predicted_tensor = model(input_tensor)
    assert predicted_tensor.size() == (batch_size, sequence_length, hidden_units), 'output has the expected dimensions'
    target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
    target_tensor.uniform_(generator=generator)
    loss = criterion(predicted_tensor, target_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
