import torch

from rau.models.transformer.unidirectional_encoder import (
    get_unidirectional_transformer_encoder
)
from rau.tasks.common.model import pad_sequences
from rau.tasks.language_modeling.vocabulary import (
    build_softmax_vocab,
    build_embedding_vocab
)
from rau.vocab import ToIntVocabularyBuilder, ToStringVocabularyBuilder

def test_unidirectional_language_model():
    to_int = ToIntVocabularyBuilder()
    to_string = ToStringVocabularyBuilder()
    vocab_content = ['a', 'b', 'c']
    softmax_to_int_vocab = build_softmax_vocab(to_int, vocab_content, allow_unk=True)
    embedding_to_int_vocab = build_embedding_vocab(to_int, softmax_to_int_vocab)
    softmax_to_string_vocab = build_softmax_vocab(to_string, vocab_content, allow_unk=True)
    embedding_to_string_vocab = build_embedding_vocab(to_string, softmax_to_string_vocab)
    kwargs = dict(
        input_vocabulary_size=len(embedding_to_string_vocab),
        output_vocabulary_size=len(softmax_to_string_vocab),
        tie_embeddings=True,
        num_layers=3,
        d_model=8,
        num_heads=4,
        feedforward_size=16,
        dropout=0.0
    )
    model_with_padding = get_unidirectional_transformer_encoder(
        **kwargs,
        use_padding=True
    )
    model_without_padding = get_unidirectional_transformer_encoder(
        **kwargs,
        use_padding=False
    )
    generator = torch.manual_seed(123)
    for (n1, p1), (n2, p2) in zip(
        model_with_padding.named_parameters(),
        model_without_padding.named_parameters(),
        strict=True
    ):
        assert n1 == n2
        p1_data = p1.data
        p1_data.uniform_(generator=generator)
        p2_data = p2.data
        if p1_data.size() != p2_data.size():
            p1_data[-1] = 0.0
            p1_data = p1_data[:-1]
        assert p1_data.size() == p2_data.size()
        p2_data.copy_(p1_data)
    batch = [
        ['a', 'b', 'c', 'a', 'c', 'b', 'c'],
        ['b', 'a', 'b', 'c', 'a', 'a'],
        ['a'],
        ['b', 'a', 'c', 'c'],
        ['c', 'b', 'a', 'a', 'c', 'b', 'a', 'b', 'c'],
        ['a', 'b', 'b'],
        ['b', 'b', 'b', 'a', 'a', 'a'],
        [],
        ['b', 'c', 'a']
    ]
    batch_as_ints = [
        torch.tensor([embedding_to_int_vocab.to_int(x) for x in example])
        for example in batch
    ]
    device = torch.device('cpu')

    def train_model(model, input_tensor, target_tensor, ignore_index):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for i in range(5):
            optimizer.zero_grad()
            logits = model(input_tensor, include_first=False)
            token_loss = torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                target_tensor,
                ignore_index=ignore_index,
                reduction='none'
            )
            sequence_loss = torch.sum(token_loss, dim=1)
            loss = torch.mean(sequence_loss, dim=0)
            loss.backward()
            optimizer.step()

    # First, we train a model that includes a separate, spurious embedding
    # vector for the padding symbol. The padding index is unique and is equal
    # to the size of the embedding vocabulary, so it tacks an extra row onto
    # the embedding matrix. Input tensors do not have EOS in them, and they
    # use the unique padding index for all padded positions. The output tensor
    # uses the same unique padding index for padded positions, which is used as
    # ignore_index for cross_entropy().
    input_without_eos = pad_sequences(
        batch_as_ints,
        device,
        pad=len(embedding_to_string_vocab),
        bos=embedding_to_string_vocab.bos_index
    )
    target_without_bos = pad_sequences(
        batch_as_ints,
        device,
        pad=len(embedding_to_string_vocab),
        eos=softmax_to_string_vocab.eos_index
    )
    train_model(
        model_with_padding,
        input_without_eos,
        target_without_bos,
        ignore_index=len(embedding_to_string_vocab)
    )

    # Next, we train a model that includes no separate embedding vector for the
    # padding symbol. We use the size of the output vocabulary as the padding
    # index. We allocate one tensor and slice it to get the input and output
    # tensors. Input tensors do have EOS in them (whose outputs should be
    # ignored and receive 0 gradient anyway).
    shared_pad = len(softmax_to_string_vocab)
    shared_tensor = pad_sequences(
        batch_as_ints,
        device,
        pad=shared_pad,
        bos=embedding_to_string_vocab.bos_index,
        eos=embedding_to_string_vocab.eos_index
    )
    input_shared = shared_tensor[:, :-1]
    target_shared = shared_tensor[:, 1:]
    train_model(
        model_without_padding,
        input_shared,
        target_shared,
        ignore_index=shared_pad
    )
    for (n1, p1), (n2, p2) in zip(
        model_with_padding.named_parameters(),
        model_without_padding.named_parameters(),
        strict=True
    ):
        assert n1 == n2
        p1_data = p1.data
        p2_data = p2.data
        if p1_data.size() != p2_data.size():
            p1_data = p1_data[:-1]
        torch.testing.assert_close(p1_data, p2_data)
