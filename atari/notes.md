# Notes on Decision Transformer

## How Decision Transformer actually works

## Atari "Engineering"


## MinGPT (Basic Transformer)
In the self attention component of the model_atari.py file, we essentially perform (q.k)v 
```python
 # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
```
Performs self attention by computing the value vector scaled using softmax. Then masks this vector to only present the 
values at a later timestep.

## Embeddings
```python
# input embedding stem
self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
# self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
self.drop = nn.Dropout(config.embd_pdrop)
```
Embeddings (positional and word) are used to numerically encode tokens, with a learned weight matrix used to generate
the token's embedding. 
