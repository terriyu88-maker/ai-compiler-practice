"""IMDB 子集数据管线：下载、解压、分词、建词表、编码为定长序列。

只依赖标准库 + torch，不依赖 torchtext，避免版本兼容问题。
"""

import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
EXTRACTED = DATA_DIR / "aclImdb"

PAD, UNK = "<pad>", "<unk>"

_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    text = _TAG_RE.sub(" ", text.lower())
    return _TOKEN_RE.findall(text)


def download_and_extract() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EXTRACTED.exists():
        if not ARCHIVE.exists():
            print(f"downloading {URL} -> {ARCHIVE}")
            urllib.request.urlretrieve(URL, ARCHIVE)
        print(f"extracting {ARCHIVE}")
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            tf.extractall(DATA_DIR)
    return EXTRACTED


def read_split(root: Path, split: str, per_class: int | None):
    samples = []
    for label, sub in ((1, "pos"), (0, "neg")):
        files = sorted((root / split / sub).glob("*.txt"))
        if per_class is not None:
            files = files[:per_class]
        for path in files:
            samples.append((path.read_text(encoding="utf-8", errors="ignore"), label))
    return samples


def build_vocab(samples, max_size: int = 20000, min_freq: int = 2) -> dict[str, int]:
    counter = Counter()
    for text, _ in samples:
        counter.update(tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for token, freq in counter.most_common():
        if len(vocab) >= max_size or freq < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)]
    if len(ids) < max_len:
        ids = ids + [vocab[PAD]] * (max_len - len(ids))
    return ids[:max_len]


class IMDBDataset(Dataset):
    def __init__(self, samples, vocab: dict[str, int], max_len: int):
        self.encoded = [(encode(t, vocab, max_len), label) for t, label in samples]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int):
        ids, label = self.encoded[index]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_datasets(
    train_per_class: int = 2500,
    test_per_class: int = 2500,
    max_len: int = 200,
    max_vocab: int = 20000,
    min_freq: int = 2,
):
    root = download_and_extract()
    train_samples = read_split(root, "train", train_per_class)
    test_samples = read_split(root, "test", test_per_class)
    vocab = build_vocab(train_samples, max_vocab, min_freq)
    return (
        IMDBDataset(train_samples, vocab, max_len),
        IMDBDataset(test_samples, vocab, max_len),
        vocab,
    )


def build_loaders(batch_size: int = 64, **kwargs):
    train_ds, test_ds, vocab = build_datasets(**kwargs)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        vocab,
    )


def main():
    train_ds, test_ds, vocab = build_datasets()
    pos = sum(1 for _, label in train_ds.encoded if label == 1)
    print(f"vocab={len(vocab)}")
    print(f"train={len(train_ds)} (pos={pos}, neg={len(train_ds) - pos})")
    print(f"test={len(test_ds)}")
    ids, label = train_ds[0]
    print(f"sample label={label.item()} len={ids.numel()} head={ids[:12].tolist()}")


if __name__ == "__main__":
    main()
