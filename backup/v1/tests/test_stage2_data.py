import torch

from src.stage2_rnn.data import PAD, UNK, build_vocab, encode, tokenize
from src.stage2_rnn.model import SentimentClassifier


def test_tokenize_strips_html_and_lowercases():
    tokens = tokenize("A <b>Great</b> Movie, really 10/10!")
    assert tokens == ["a", "great", "movie", "really", "10", "10"]


def test_vocab_and_encode_padding():
    samples = [("good good movie", 1), ("bad movie", 0)]
    vocab = build_vocab(samples, max_size=10, min_freq=1)
    assert vocab[PAD] == 0 and vocab[UNK] == 1
    ids = encode("good unknown", vocab, max_len=5)
    assert len(ids) == 5
    assert ids[0] == vocab["good"]
    assert ids[1] == vocab[UNK]
    assert ids[2:] == [vocab[PAD]] * 3


def test_model_shapes_for_both_cells():
    model = SentimentClassifier(vocab_size=50, embed_dim=8, hidden_dim=8, cell_type="lstm")
    x = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    logits = model(x)
    assert logits.shape == (2, 2)
