import torch
import torch.nn as nn
from torch.nn import functional as F

# Params
block_size = 8  # chunk of size 8
batch_size = 4  # get more blocks at a time to efficiently use gpu
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ---------


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all the characters used in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# develop a strategy to tokenize the input text
# tokenize means to convert the text in some sequence of meaningful integers
# according to possible vocabulary of elements
# we will use enumerate to achieve this, and the order followed will be what we got from sorted above i.e chars

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# making functions to achieve encoding and decoding
encode = lambda s: [stoi[c] for c in s]  # take in string returns list of integers
decode = lambda lis: ''.join([itos[i] for i in lis])  # takes in list of int and returns string

# the above tokenizers can be changed like tiktoken(OPENAI) and BPE(GOOGLE)

data = torch.tensor(encode(text), dtype=torch.long)

# divide the data in train and test set
n = int(0.9 * len(data))
train = data[:n]
test = data[n:]

# we will not train the transformer with the whole dataset at once
# it is very computation heavy
# we will divide the data into chunks

torch.manual_seed(1337)


def get_batch(split):
    data = train if split == "train" else test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model.forward(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


xb, yb = get_batch('train')

# one of the simplest word prediction models given a word sequence is Bigram Model.
# Implementing Bigram Model

torch.manual_seed(1337)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # pytorch will arrange it in (batch, time, channel) tensor

        # batch = batch_size, time = block_size, channel = vocab_size
        # the cross_entropy expects the logits dimension to be (B,C) and not (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # stretch the tensor making it 2d
            targets = targets.view(B * T)  # making this 1d
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, loss = self.forward(idx)
            logits = logits[:, -1, :]  # converts logits to (B,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # give the next word prediction
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramModel(vocab_size)

# create a pytorch optimization model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # get batch
    xb, yb = get_batch('train')

    # print avg loss
    '''if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss is: {losses['train']:.4f}, test loss is {losses['test']:.4f}")'''
    # eval the loss
    logit, loss = model.forward(xb, yb)  # giving call to model.forward to give logit and loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=50)[0].tolist()))


# our self attention block
# the mathematical trick in self-attention

# example starts
B,T,C = 4,8,2
x = torch.randn(B,T,C)
print(x.shape)
