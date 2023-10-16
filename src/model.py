
#import the librairies
import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------- Head --------------# 

class Head(nn.Module): 
    """
    Defines a single attention
    Args:
        cfg (dict): Configuration parameters.
        head_size (int): Size of the head =>  head_size = d_k = d_v if we are used to the notation of the original paper "Attention is all you need"
    """

    # For the forward methods:
    # Input of size (batch, time-step, channels)
    # Output of size (batch, time-step, head size)

    def __init__(self,cfg, head_size):  
        super().__init__()

         # Linear transformations (just matrix multiplication without even a bias) for keys, queries, and values
        self.values = nn.Linear(cfg['n_embed'], head_size, bias=False) 
        self.keys = nn.Linear(cfg['n_embed'], head_size, bias=False)
        self.queries = nn.Linear(cfg['n_embed'], head_size, bias=False)
        
        # for the self mask attention
        self.register_buffer('tril', torch.tril(torch.ones(cfg['block_size'], cfg['block_size'])))
        self.dropout = nn.Dropout(cfg['dropout']) #we use dropout in order to prevent/reduce overfitting 


    def forward(self, x):
        # x.shape: (batch, time-step, channels)
        B, T, C = x.shape
        key = self.keys(x)  # (B, T, head_size)
        query = self.queries(x)  # (B, T, head_size)
        value = self.values(x)  # same
        energy = query @ key.transpose(-2, -1) # compute the dot product between QK^T
        energy = energy / (query.shape[-1] ** 0.5)  # (B, T, T)
        energy = energy.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #fill the mask part by -inf so that after the softmax it will output 0 

        energy = F.softmax(energy, dim=-1)
        energy = self.dropout(energy) 
        out = energy @ value  # (B, T, C)
        return out



# --------------- Multi Head --------------# 

class MultiHeadAttention(nn.Module):
    """
    Using the head class, it implements the multi_head attention.

    The idea is to use different head, so that each head can specialize in something different. 
    And at the end we concatenate the result so that we have a more complete embedding

    Args:
        cfg (dict): Configuration parameters.
        head_size (int): Size of each attention head.

    """
    def __init__(self, cfg, head_size):
        super().__init__()
        # Create a list of 'num_heads' attention heads
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg['num_heads'])])
        self.proj = nn.Linear(cfg['n_embed'], cfg['n_embed'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1) #concatenation of the result of the different heads, dim = (B, T, head_size*num_head = n_embed)
        out = self.proj(x) 
        out = self.dropout(out)
        return out


# --------------- Feedforward --------------# 
class FeedForward(nn.Module): 
    def __init__(self, cfg):
        super().__init__()

        # The first linear layer increases the dimensionality by a factor of 4, same factor described in the original paper
        self.linear = nn.Linear(cfg['n_embed'], 4 * cfg['n_embed']) 
        self.linear2 = nn.Linear(4 * cfg['n_embed'], cfg['n_embed']) # Projecting back to the original embedding size
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# --------------- Block --------------# 

class Block(nn.Module):
    """"
    Putting all together to build the transformer (decoder) block 

    Args:
        cfg (dict): Configuration parameters.

    """
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg['n_embed'] // cfg['num_heads'] # headsize is 
        self.sa = MultiHeadAttention(cfg, head_size)
        self.ffn = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg['n_embed']) 
        self.ln2 = nn.LayerNorm(cfg['n_embed']) 

    def forward(self, x):
        x_sa = self.sa(x)
        x_ffn = self.ffn(x)
        x = x + x_sa  # here we use skip connection (meaning that add the input of one part of the nn to the output)
        x = x + x_ffn # it helps when we calculate the gradient during the training loop
        return x


# --------------- GPT Model --------------# 

class GPTmodel(nn.Module):
    """
    Defines the GPT model from one end to the other

    Args:
        cfg (dict): Configuration parameters.
        vocab_size (int): Size of the vocabulary.

    Methods:
        forward(idx, targets): Forward pass for the GPT model.
        generate(idx, max_new_tok): Generates new text using the GPT model.
    """
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
        tok_embed = self.token_embedding(idx) # we associate each word an embedding vector, embedding learned throught the training
        pos_embed = self.position_embedding(torch.arange(T, device=self.cfg['device'])) # same for the the position embedding
        x = pos_embed + tok_embed # we will feed to the attention block the sum of those 2 previous embeddings
        logits = self.attention_blocks(x) 
        logits = self.linear(logits) # a last linear transformation to have an ouput of the size of the vocabulary

        # if there is a target (pass as argument), we want to compute and return the loss
        if targets is None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # the cross entropy loss expect an input of the shape Batch * Channel
            targets = targets.view(B * T) # and for the target just a one dimension tensor
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tok):

        for i in range(max_new_tok):
            #the context cannot extend the block_size otherwise it will generate an error when we embded the position so we crop it if needed
            idx_crop = idx[:, -self.cfg['block_size']:] 
            logits, loss = self(idx_crop)

            logits = logits[:, -1, :] #we take the last result because it represent the word to predict
            probs = F.softmax(logits, dim=-1) # we apply a softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # then we use thoses probabilities to choose the next word according to those probs

            idx = torch.cat((idx, idx_next), dim=1) #we add the index of the word we predict in order for it to be use as context for next predictions

        return idx
