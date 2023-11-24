"""Evaluate the model by calculating its micro/macro F1 score and accuracy."""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from agnews_dataset import AGNewsDataset, preprocess_data
from model.lstm import LSTMClassifier

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    arg_parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.abspath(
            os.path.join(__file__, os.pardir, os.pardir, "data", "test.csv")
        ),
    )
    arg_parser.add_argument("--model_path", type=str, required=True)
    args = arg_parser.parse_args()

    model_ckpt = torch.load(args.model_path, map_location=args.device)

    # Load the test data
    text, label = preprocess_data(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=model_ckpt["model_args"]["max_length"],
    )
    dataset = AGNewsDataset(encoding, label)
    dataloader = DataLoader(dataset, batch_size=32)

    # Load the model
    model = LSTMClassifier(
        vocab_size=len(tokenizer),
        embedding_dim=model_ckpt["model_args"]["embedding_dim"],
        hidden_dim=model_ckpt["model_args"]["hidden_dim"],
        num_layers=model_ckpt["model_args"]["num_layers"],
        output_dim=model_ckpt["model_args"]["output_dim"],
        dropout=model_ckpt["model_args"]["dropout"],
    ).to(args.device)
    model.load_state_dict(model_ckpt["state_dict"])

    # Evaluate the model
    model.eval()
    preds = []
    for batch in dataloader:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(args.device)
            logits = model(input_ids)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    label = np.array(label)

    print(f"Accuracy: {accuracy_score(label, preds)}")
    print(f"Macro F1: {f1_score(label, preds, average='macro')}")
    print(f"Micro F1: {f1_score(label, preds, average='micro')}")
