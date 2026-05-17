from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


SAMPLE_CORPUS = """
Architecting intelligence for AI systems means turning raw text into representations, learning
patterns over sequences, and evaluating whether a model generalizes beyond the
examples it has memorized. A small character-level model cannot match a modern
transformer, but it is a useful laboratory for embeddings, recurrence, hidden
state, loss curves, and sampling temperature.
"""


@dataclass(frozen=True)
class TrainConfig:
    embed_size: int = 64
    hidden_size: int = 128
    n_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 40
    batch_size: int = 16
    lr: float = 0.003
    steps: int = 25
    seed: int = 42


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output.reshape(-1, output.size(-1)))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            weight.new_zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )


def load_text(path: str | Path | None = None) -> str:
    if path is None:
        return SAMPLE_CORPUS
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input text file is empty: {path}")
    return text


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    if "<UNK>" not in chars:
        chars.append("<UNK>")
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char


def encode(text: str, char_to_idx: dict[str, int]) -> torch.Tensor:
    unknown = char_to_idx["<UNK>"]
    return torch.tensor([char_to_idx.get(char, unknown) for char in text], dtype=torch.long)


def get_batch(data: torch.Tensor, seq_len: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= seq_len + 1:
        raise ValueError("Text is too short for the requested sequence length.")
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[start : start + seq_len] for start in starts])
    y = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)


def train_model(text: str, config: TrainConfig = TrainConfig()) -> tuple[CharLSTM, dict[str, object]]:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chars, char_to_idx, idx_to_char = build_vocab(text)
    data = encode(text, char_to_idx)
    model = CharLSTM(
        vocab_size=len(chars),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    model.train()
    for _ in range(config.steps):
        x, y = get_batch(data, config.seq_len, config.batch_size, device)
        hidden = model.init_hidden(config.batch_size, device)
        optimizer.zero_grad()
        logits, _ = model(x, hidden)
        loss = criterion(logits, y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(float(loss.item()))

    metadata: dict[str, object] = {
        "device": str(device),
        "vocab_size": len(chars),
        "characters": len(text),
        "losses": losses,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }
    return model, metadata


@torch.no_grad()
def generate_text(
    model: CharLSTM,
    start: str,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    length: int = 120,
    temperature: float = 0.8,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = next(model.parameters()).device
    model.eval()
    hidden = model.init_hidden(1, device)
    unknown = char_to_idx["<UNK>"]
    generated = start
    input_ids = torch.tensor([[char_to_idx.get(char, unknown) for char in start]], dtype=torch.long, device=device)

    for position in range(max(0, input_ids.size(1) - 1)):
        _, hidden = model(input_ids[:, position : position + 1], hidden)

    current = input_ids[:, -1:]
    unknown_id = char_to_idx.get("<UNK>")
    for _ in range(length):
        logits, hidden = model(current, hidden)
        next_logits = logits[-1].clone()
        if unknown_id is not None:
            next_logits[unknown_id] = -torch.inf
        probs = F.softmax(next_logits / temperature, dim=0)
        next_id = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_id]
        generated += next_char
        current = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated


def run_demo() -> dict[str, object]:
    config = TrainConfig(steps=5, batch_size=8, seq_len=32)
    model, metadata = train_model(load_text(), config)
    generated = generate_text(
        model,
        start="AI",
        char_to_idx=metadata["char_to_idx"],  # type: ignore[arg-type]
        idx_to_char=metadata["idx_to_char"],  # type: ignore[arg-type]
        length=40,
    )
    losses = metadata["losses"]  # type: ignore[assignment]
    return {
        "vocab_size": metadata["vocab_size"],
        "characters": metadata["characters"],
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "sample": generated,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
