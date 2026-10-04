from torch import nn

CELL_TYPES = ("rnn", "lstm")


class SentimentClassifier(nn.Module):
    """Embedding + 循环层 + 掩码平均池化 + 线性分类头。

    cell_type 取 "rnn" 或 "lstm"，其余结构完全一致，用于公平对比两种循环单元。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        cell_type: str = "rnn",
        num_classes: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        if cell_type not in CELL_TYPES:
            raise ValueError(f"cell_type must be one of {CELL_TYPES}, got {cell_type}")
        self.cell_type = cell_type
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        rnn_cls = nn.RNN if cell_type == "rnn" else nn.LSTM
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != self.pad_idx).unsqueeze(-1).to(x.dtype)
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.fc(self.dropout(pooled))
