
#import the librairies
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# --------------- Head --------------# 

class Head(nn.Module):
    # For the forward methods:
    # Input of size (batch, time-step, channels)
    # Output of size (batch, time-step, head size)

    def __init__(self, head_size):  # head_size = d_k = d_v if we are used to the notation of the original paper
        super().__init__()
        self.values = nn.Linear(cfg['n_embed'], head_size, bias=False)
        self.keys = nn.Linear(cfg['n_embed'], head_size, bias=False)
        self.queries = nn.Linear(cfg['n_embed'], head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(cfg['block_size'], cfg['block_size'])))
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        # x.shape: (batch, time-step, channels)
        B, T, C = x.shape
        key = self.keys(x)  # (B, T, head_size)
        query = self.queries(x)  # (B, T, head_size)
        value = self.values(x)  # same
        energy = query @ key.transpose(-2, -1)
        energy = energy / (query.shape[-1] ** 0.5)  # (B, T, T)
        energy = energy.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        energy = F.softmax(energy, dim=-1)
        energy = self.dropout(energy)
        out = energy @ value  # (B, T, C)
        return out



# --------------- Multi Head --------------# 

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(cfg['num_heads'])])
        self.proj = nn.Linear(cfg['n_embed'], cfg['n_embed'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(x)
        out = self.dropout(out)
        return out


# --------------- Feedforward --------------# 
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg['n_embed'], 4 * cfg['n_embed'])
        self.linear2 = nn.Linear(4 * cfg['n_embed'], cfg['n_embed'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# --------------- Block --------------# 

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg['n_embed'] // cfg['num_heads']
        self.sa = MultiHeadAttention(cfg, head_size)
        self.ffn = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg['n_embed'])
        self.ln2 = nn.LayerNorm(cfg['n_embed'])

    def forward(self, x):
        x_sa = self.sa(x)
        x_ffn = self.ffn(x)
        x = x + x_sa
        x = x + x_ffn
        return x


# --------------- GPT Model --------------# 

class GPTmodel(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg['n_embed'])
        self.position_embedding = nn.Embedding(cfg['block_size'], cfg['n_embed'])
        self.attention_blocks = nn.Sequential(
            *[Block(cfg) for _ in range(cfg['num_blocks'])],
            nn.LayerNorm(cfg['n_embed']),
        )
        self.linear = nn.Linear(cfg['n_embed'], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(torch.arange(T, device=self.cfg['device']))
        x = pos_embed + tok_embed
        logits = self.attention_blocks(x)
        logits = self.linear(logits)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tok):
        for i in range(max_new_tok):
            idx_crop = idx[:, -self.cfg['block_size']:]
            logits, loss = self(idx_crop)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx



