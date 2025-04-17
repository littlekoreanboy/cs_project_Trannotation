import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import DNATokenizer
from dataset import DNASequenceDataset, load_and_prepare_data
from model import DNATransformer
import argparse
import csv

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device).float()

        optimizer.zero_grad()
        logits = model(sequences).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()

        if torch.isnan(loss):
            print("NaN loss encountered")
            print("Logits:", logits)
            print("Labels:", labels)

        # Optional gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        preds = torch.sigmoid(logits) > 0.5
        correct += (preds == labels.bool()).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device).float()
            logits = model(sequences).squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total

def user_sequence(model, model_path, tokenizer, sequence, seq_len, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\nOriginal Sequence:\n{sequence}")

    # Tokenize the sequence
    tokenized = tokenizer.tokenize(sequence)
    token_ids = tokenizer.encode(sequence, max_length=seq_len)

    # Pad or truncate
    if len(token_ids) < seq_len:
        token_ids += [tokenizer.pad_token_id] * (seq_len - len(token_ids))
    else:
        token_ids = token_ids[:seq_len]

    print(f"\nTokenized Sequence (IDs):\n{token_ids}")

    # Convert to tensor
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(token_tensor).squeeze(0)
        prob = torch.sigmoid(logits).item()
        label = "gene" if prob > 0.5 else "non-gene"

    print(f"\nPrediction: {label}")
    print(f"Probability: Gene = {prob:.4f}, Non-Gene = {1 - prob:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="train or test mode")
    parser.add_argument('--data_path', type=str, help="Path to the training dataset (required for training)")
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)

    parser.add_argument("--model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--user_sequence", type=str, help="DNA sequence for testing")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DNATokenizer(k=args.k, vocab_file=args.vocab_path)

    if args.mode == "train":
        if not args.data_path:
            raise ValueError("You must provide --data_path for training mode")

        train_loader, val_loader = load_and_prepare_data(args.data_path, args.batch_size, tokenizer, args.seq_len)

        model = DNATransformer(
            d_model=args.d_model,
            vocab_size=len(tokenizer.vocab),
            seq_len=args.seq_len,
            dropout=args.dropout,
            h=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(args.epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Save model
        torch.save(model.state_dict(), "trained_model.pth")

        # Save training log
        with open("training_log.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
            for i in range(args.epochs):
                writer.writerow([
                    i + 1,
                    train_losses[i],
                    train_accuracies[i],
                    val_losses[i],
                    val_accuracies[i]
                ])

    elif args.mode == "test":
        if not args.model_path or not args.user_sequence:
            raise ValueError("You must provide both --model_path and --user_sequence for testing mode")

        model = DNATransformer(
            d_model=args.d_model,
            vocab_size=len(tokenizer.vocab),
            seq_len=args.seq_len,
            dropout=args.dropout,
            h=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers
        )

        user_sequence(
            model=model,
            model_path=args.model_path,
            tokenizer=tokenizer,
            sequence=args.user_sequence,
            seq_len=args.seq_len,
            device=device
        )

if __name__ == '__main__':
    main()
