import json
from itertools import product

class DNATokenizer:
    def __init__(self, k=None, vocab_file=None, pad_token='<pad>', unk_token='<unk>'):
        self.k = k
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = {self.pad_token: 1, self.unk_token: 0}
        self.reverse_vocab = {1: self.pad_token, 0: self.unk_token}

        if vocab_file:
            self.load_vocab(vocab_file)
        elif self.k is not None:
            self.build_vocab()

    def build_vocab(self, sequences=None):
        if sequences:
            for seq in sequences:
                seq = seq.upper()
                for i in range(len(seq) - self.k + 1):
                    kmer = seq[i:i+self.k]
                    if kmer not in self.vocab:
                        index = len(self.vocab)
                        self.vocab[kmer] = index
                        self.reverse_vocab[index] = kmer
        else:
            nucleotides = ['A', 'C', 'G', 'T']
            all_kmers = [''.join(p) for p in product(nucleotides, repeat=self.k)]
            for kmer in all_kmers:
                if kmer not in self.vocab:
                    index = len(self.vocab)
                    self.vocab[kmer] = index
                    self.reverse_vocab[index] = kmer

    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, filepath):
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        kmer_keys = [k for k in self.vocab if k not in (self.pad_token, self.unk_token)]

        if kmer_keys:
            self.k = len(kmer_keys[0])
        else:
            raise ValueError("Vocabulary file does not contain valid k-mers.")

    def tokenize(self, sequence, stride=None):
        stride = stride or self.k
        return [sequence[i:i+self.k] for i in range(0, len(sequence) - self.k + 1, stride)]

    def encode(self, sequence, max_length=512):
        sequence = sequence.upper()
        tokens = self.tokenize(sequence)
        token_ids = [self.vocab.get(kmer, self.vocab[self.unk_token]) for kmer in tokens]
        if len(token_ids) < max_length:
            token_ids += [self.vocab[self.pad_token]] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids):
        return [self.reverse_vocab.get(tid, self.unk_token) for tid in token_ids]

#tokenizer = DNATokenizer(k=6)
#tokenizer.build_vocab()  # No sequences passed, builds full 4096 vocab
#tokenizer.save_vocab("vocab_6kmer.json")

#tokenizer = DNATokenizer(k=6)
#sequence = "ACGTACGTACGTACGTACGTA"  # test example
#tokens = tokenizer.tokenize(sequence)
#ids = tokenizer.encode(sequence)

#print("Original:", sequence)
#print("Tokens:", tokens)
#print("Token IDs:", ids)