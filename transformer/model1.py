import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.rand(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    

class Layernormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self , x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class ResidualConnection(nn.Module):
    def __init__(self, feature:int , dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Layernormalization(feature)

    def forward(self, x, sublayers):
        return x + self.dropout(sublayers(self.norm(x)))
        
class MultiheadAttentionBlock(nn.Module):
    def __init__(self,d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divide by h"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
    
    @staticmethod
    def attention(query, key, value, mask, drouput:nn.Dropout):
        d_k = query[-1]
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if drouput is not None:
            attention_scores = drouput(attention_scores)
        return (attention_scores @ value) , attention_scores
    

    def forward(self,q,k,v, mask):
        query = self.w_k(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h,seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # calculate a attention
        x , self.attention_scores = MultiheadAttentionBlock.attention(query, key,value,mask, self.dropout)

        # If x has shape [batch_size, num_heads, seq_length, d_k], then after transposing dimensions 1 and 2, the shape becomes [batch_size, seq_length, num_heads, d_k].
        x = x.transpose(1,2).contigous.view(x.shape(0), -1, self.h*self.self.d_k)
        return self.w_o(x)
class ResidualConnectional(nn.modules):
    def __init__(self, feature:int, dropout:float):
        self.dropout = nn.Dropout(dropout)
        self.norm = Layernormalization(feature)

    def forward(self , x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiheadAttentionBlock, feed_forward_block:FeedForward, feature:int, dropout:float):
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual = nn.ModuleList(ResidualConnectional(feature, dropout) for _ in range(2))
    def forward(self, x, src_mask):
        x = self.residual[0](x , lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual[1](x , self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,feature:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Layernormalization(feature)

    def forward(self, x,src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiheadAttentionBlock, cross_attention_block:MultiheadAttentionBlock, feed_forward_block:FeedForward, feature:int, dropout:float):
        self.self_attention_block = self_attention_block
        self.cross_attention_block = self.cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual = nn.ModuleList(ResidualConnectional(feature, dropout) for _ in range(2))
    def forward(self, x,encoder_output, src_mask, tgt_mask):
        x = self.residual[0](x , lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual[1](x , lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual[2](x , self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self,feature:int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Layernormalization(feature)

    def forward(self, x , encoder_output,src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x , encoder_output, src_mask , tgt_mask)
        return self.norm(x)

class Projection_layer(nn.Module):
    def __init__(self, d_model , vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (B,S,D)--> (B , S , Vocab)
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_emb:InputEmbedding, tgt_emb : InputEmbedding , src_pos:PositionalEncoding, tgt_pos:PositionalEncoding , proj:Projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj
    
    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src , src_mask)

    def decode(self,encoder_output:torch.Tensor, src_mask:torch.Tensor,tgt:torch.Tensor, tgt_mask:torch.Tensor):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj(x)
    