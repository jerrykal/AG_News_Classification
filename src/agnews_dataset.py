import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer


class AGNewsDataset(Dataset):
    def __init__(self, encoding: BatchEncoding, label: list) -> None:
        self.encoding = encoding
        self.label = label

    def __getitem__(self, idx: int) -> dict:
        sample = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        sample["labels"] = torch.tensor(self.label[idx])
        return sample

    def __len__(self) -> int:
        return len(self.label)


def preprocess_data(
    data_path: str,
    use_news_title: bool = True,
) -> tuple[list, list]:
    """Preprocess data for AG News dataset."""
    # Read data
    df = pd.read_csv(data_path)
    text = (
        df["Title"] + df["Description"] if use_news_title else df["Description"]
    ).tolist()
    label = (df["Class Index"] - 1).tolist()

    return (text, label)
