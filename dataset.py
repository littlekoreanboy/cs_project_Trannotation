import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import DNATokenizer

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, sequence_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer.encode(seq, self.sequence_length)
        return torch.tensor(tokenized, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def preprocess_sequence(sequence, tokenizer, sequence_length):
    return tokenizer.encode(sequence, sequence_length)

def load_and_prepare_data(file, batch_size, tokenizer, sequence_length, val_split=0.2):
    sequences, labels = [], []

    with open(file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                label, seq = line.strip().split(',')
                sequences.append(seq.upper())
                labels.append(int(label))
            except Exception as e:
                print(f"Skipping line due to error: {line.strip()} — {e}")

    dataset = DNASequenceDataset(sequences, labels, tokenizer, sequence_length)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Loaded {len(train_dataset)}. Loaded {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
