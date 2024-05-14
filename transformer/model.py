import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# class InputEmbedding(nn.Module):
#     def init(self, d_model:int, vocab_size:int):
#         super().__init__()
#         self.d_model = d_model
#         self.vocal_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, x):
#         return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len , d_model) 
        pe = torch.rand(seq_len , d_model)
        # create a vector of shape seq_len
        positional = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        # create a vector of shape d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # apply a sine to even indices
        pe[:,0::2] = torch.sin(positional * div_term)
        # apply a cosine to odd indices
        pe[:,1::2] = torch.cos(positional * div_term)
        # add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the positional encoding as buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias

class FeedForwardblock(nn.Module):
    def __init__(self, d_model:int, d_ff:int , dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    



class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divided by h"
        self.dropout = nn.Dropout(dropout)

        d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(self, query, key, value, mask, dropout:float):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h , seq_len, seq_len)
        return (attention_scores @ value) , attention_scores


    def forward(self, q , k , v , mask):
        query = self.w_q(q) # (batch, seq_len , d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len , d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len , d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1] , self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1] , self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1] , self.h, self.d_k).transpose(1,2)

        # Calculate attention
        x , self.attention_scores = MultiheadAttentionBlock.attention(query, key, value , mask)

        # (Batch, seq_len, d_k) --> (Batch, seq_len, h , d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    
class Residual_Connetion(nn.Module):
    def __init__(self, dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiheadAttentionBlock , feed_forward_block:FeedForwardblock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections=nn.ModuleList([Residual_Connetion(dropout) for _ in range(2)])

    def forward(self, x , src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x,x,x , src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x , src_mask)
        return self.norm(x)