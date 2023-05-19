import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all the characters used in the text

chars = sorted(list(set(text)))

# develop a strategy to tokenize the input text
# tokenize means to convert the text in some sequence of meaningful integers
# according to possible vocabulary of elements
# we will use enumerate to achieve this, and the order followed will be what we got from sorted above i.e chars

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# making functions to achieve encoding and decoding
encode = lambda s: [stoi[c] for c in s]        # take in string returns list of integers
decode = lambda lis: ''.join([itos[i] for i in lis])        # takes in list of int and returns string

# the above tokenizers can be changed like tiktoken(OPENAI) and BPE(GOOGLE)


data = torch.tensor(encode(text), dtype=torch.long)


# divide the data in train and test set
n = int(0.9*len(data))
train = data[:n]
test = data[n:]

# we will not train the transformer with the whole dataset at once
# it is very computation heavy
# we will divide the data into chunks 

block_size = 8
batch_size = 4    # get more blocks at a time to efficiently use gpu 
torch.manual_seed(1337)

def get_batch(split):
  data = train if split =="train" else test
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  return x,y

xb, yb = get_batch('train')
print(xb)
print(yb)