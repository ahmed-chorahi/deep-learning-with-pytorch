# Lesson 7: The Transformer encoder
# Encoder = word embedding + positional encoding + N Transformer blocks.
# It turns a sentence into one context-aware vector per word.

import math
import torch
import torch.nn as nn

torch.manual_seed(0)


def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


# Same block as in 06_transformer_block (copied so this lesson runs on its own)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_heads=4, d_ff=64, num_layers=2, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))   # saved, but not learned
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, ids):
        pad_mask = ids == 0                                          # True where the word is <pad>
        x = self.embedding(ids) * math.sqrt(self.d_model)            # scale the embeddings
        x = x + self.pe[:ids.shape[1]]                               # add positions
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        return x


encoder = Encoder(vocab_size=100)

print("=== 1. The encoder ===")
print("blocks:", len(encoder.blocks), "| d_model:", encoder.d_model)
print("parameters:", sum(p.numel() for p in encoder.parameters()))

print("\n=== 2. Running a batch (0 = <pad>) ===")
ids = torch.tensor([[5, 8, 2, 9, 0, 0],
                    [7, 3, 4, 6, 1, 2]])
out = encoder(ids)
print("input ids :", tuple(ids.shape), "= (batch, words)")
print("output    :", tuple(out.shape), "= (batch, words, d_model)")

print("\n=== 3. The padding mask in action ===")
attn = nn.MultiheadAttention(8, 2, batch_first=True)
x = torch.randn(1, 4, 8)
mask = torch.tensor([[False, False, False, True]])         # last word is padding
_, w = attn(x, x, x, key_padding_mask=mask)
print("attention given to the padded word:", w[0, :, -1].tolist())

print("\n=== 4. Padding does not change real words ===")
encoder.eval()
with torch.no_grad():
    short = encoder(torch.tensor([[5, 8, 2]]))
    padded = encoder(torch.tensor([[5, 8, 2, 0, 0]]))
print("same vectors for the real words?", torch.allclose(short[0], padded[0, :3], atol=1e-4))

print("\n=== 5. PyTorch also has a ready-made encoder ===")
layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
builtin = nn.TransformerEncoder(layer, num_layers=2)
print("built-in output shape:", tuple(builtin(torch.randn(2, 6, 32)).shape))

print("\n=== 6. Pooling: one vector for the whole sentence ===")
sentence_vector = out.mean(dim=1)
print("mean over words:", tuple(sentence_vector.shape), "= (batch, d_model)")
