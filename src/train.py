import argparse
import os
import random
import time

import numpy
import torch
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agnews_dataset import AGNewsDataset, preprocess_data
from model.lstm import LSTMClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    arg_parser = argparse.ArgumentParser()

    # Training arguments
    arg_parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    arg_parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.abspath(
            os.path.join(__file__, os.pardir, os.pardir, "data", "train.csv")
        ),
    )
    arg_parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "model")),
    )
    arg_parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "logs")),
    )
    arg_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Data arguments
    arg_parser.add_argument("--use_news_title", action="store_true")

    # Model arguments
    arg_parser.add_argument(
        "--model", type=str, choices=["lstm", "bert"], default="lstm"
    )
    arg_parser.add_argument("--max_length", type=int, default=128)
    arg_parser.add_argument("--embedding_dim", type=int, default=256)
    arg_parser.add_argument("--hidden_dim", type=int, default=256)
    arg_parser.add_argument("--num_layers", type=int, default=1)
    arg_parser.add_argument("--dropout", type=float, default=0.5)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--num_epochs", type=int, default=20)
    arg_parser.add_argument("--lr", type=float, default=1e-3)

    return arg_parser.parse_args()


def fix_seed(seed: int) -> None:
    """Fix random seed for reproducibility."""
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)


def train(args: argparse.Namespace) -> None:
    """Train a model for text classification."""
    save_name = f"{args.model}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    save_path = os.path.join(args.save_dir, save_name + ".pt")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, save_name))

    # Fix random seed for reproducibility
    fix_seed(args.seed)

    # Load data
    text, label = preprocess_data(args.data_path, args.use_news_title)

    # Split dataset into training and validation sets
    train_text, val_text, train_labels, val_labels = train_test_split(
        text, label, test_size=0.2, random_state=args.seed
    )

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_encoding = tokenizer(
        train_text,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    val_encoding = tokenizer(
        val_text,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )

    # Create dataset
    train_dataset = AGNewsDataset(train_encoding, train_labels)
    val_dataset = AGNewsDataset(val_encoding, val_labels)

    # Create dataloader
    num_classes = len(set(train_labels))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    if args.model == "lstm":
        model = LSTMClassifier(
            vocab_size=len(tokenizer),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=num_classes,
            dropout=args.dropout,
        ).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_classes,
        ).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create optimizer & loss function
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # Train
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            if args.model == "lstm":
                outputs = model(batch["input_ids"])
                loss = criterion(outputs, batch["labels"])
            else:
                outputs = model(**batch)
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            logits = outputs if args.model == "lstm" else outputs.logits
            train_acc += (logits.argmax(1) == batch["labels"]).sum().item()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataset)

        writer.add_scalar("Train/Acc", train_loss, epoch)
        writer.add_scalar("Train/Loss", train_acc, epoch)

        # Validation
        model.eval()
        for batch in val_dataloader:
            with torch.no_grad():
                batch = {k: v.to(args.device) for k, v in batch.items()}

                if args.model == "lstm":
                    outputs = model(batch["input_ids"])
                    loss = criterion(outputs, batch["labels"])
                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                val_loss += loss.item()

                logits = outputs if args.model == "lstm" else outputs.logits
                val_acc += (logits.argmax(1) == batch["labels"]).sum().item()

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataset)

        writer.add_scalar("Val/Acc", val_loss, epoch)
        writer.add_scalar("Val/Loss", val_acc, epoch)

        print(
            f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Saving the best model along with its hyperparameters
        if val_acc > best_acc:
            print(f"Saving model with val acc: {val_acc:.4f}")
            best_acc = val_acc
            torch.save(
                {
                    "model_args": {
                        "seed": args.seed,
                        "use_news_title": args.use_news_title,
                        "max_length": args.max_length,
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "output_dim": num_classes,
                        "dropout": args.dropout,
                    },
                    "state_dict": model.state_dict(),
                },
                save_path,
            )

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
