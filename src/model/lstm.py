import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """News classifier using LSTM"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedding = self.dropout(self.embedding(input))
        _, (hidden, _) = self.lstm(embedding)

        # Extract the last embedding
        hidden = self.dropout(hidden[-1, :, :])

        out = self.fc(hidden)
        return out
