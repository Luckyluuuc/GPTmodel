import torch
import re

#-----------------

def tokenize_text(text):
    # Use a regular expression to tokenize the text into words while preserving spaces and line breaks
    tokens =  re.findall(r'\w+|[\.,;!?"]+|\n|\t|.', text)
    return tokens
#-----------------
def mapp(text): # return 2 dictionnaries mapping each token to an integers and vice versa
  list_token = sorted(list(set(text)))
  vocab_size = len(list_token)
  stoi = {c:i for i,c in enumerate(list_token)}
  itos = {i:c for i,c in enumerate(list_token)}

  return vocab_size, stoi, itos
#-----------------

def splitting_data(frac, data):
    z = int(frac*len(data))
    train_set = data[:z]
    validation_set = data[z:]
    return train_set, validation_set
  
#-----------------

def encode(text, stoi):
  list_integers = []
  for c in text:
    list_integers.append(stoi.get(c))

  return list_integers

#-----------------

def decode(list_integers, itos):
  text = []
  for i in list_integers:
    text.append(itos.get(i))

  text = ''.join(c for c in text) #delete this line if you want a list of char instead of a str
  return text
    
#-----------------

def get_batch(split, block_size, batch_size, device, train_set, validation_set): #split is either "train" or "eval"
  assert split in ["train", "eval"], "split must be 'train' or 'eval'"
  data = train_set if split == "train" else validation_set

  ix = torch.randint(0, len(data) - block_size-1, (batch_size,)) # return a tensor of shape (batch_size) with random values between 0 and len(data) - block_size

  x = torch.stack([torch.tensor(data[i:i + block_size]) for i in ix])
  y = torch.stack([torch.tensor(data[i + 1:i + block_size + 1]) for i in ix])

  x, y = x.to(device), y.to(device)
  return x, y
#-----------------

@torch.no_grad()
def estimate_loss(m, eval_iters, train_set, evalutation_set, block_size, batch_size, device):
  out = {}
  m.eval()
  for split in ["train", "eval"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split, block_size, batch_size, device, train_set, evalutation_set)
      logits, loss = m(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  m.train()
  return out
#-----------------

class WeightManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_weights(self, model):
        torch.save(model.state_dict(), self.file_path)

    def load_weights(self, model):
        model.load_state_dict(torch.load(self.file_path))
        model.eval()  # Set the model to evaluation mode
