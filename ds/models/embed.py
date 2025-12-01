import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class CurveEmbedding(nn.Module):
    def __init__(self, num_types, vocab_size, d_model):
        super(CurveEmbedding, self).__init__()
        self.type_embedding = nn.Embedding(num_types, d_model)
        self.param_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: [Batch, SeqLen, 5]
        # x[:, :, 0] is curve type
        # x[:, :, 1:] are params
        
        curve_types = x[:, :, 0]
        params = x[:, :, 1:]
        
        type_embed = self.type_embedding(curve_types) # [Batch, SeqLen, d_model]
        param_embeds = self.param_embedding(params)   # [Batch, SeqLen, 4, d_model]
        
        # Sum parameter embeddings
        param_embed_sum = torch.sum(param_embeds, dim=2) # [Batch, SeqLen, d_model]
        
        return type_embed + param_embed_sum

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, vocab_size=None, num_types=None):
        super(DataEmbedding, self).__init__()

        self.embed_type = embed_type
        if embed_type == 'curve':
            if vocab_size is None or num_types is None:
                raise ValueError("vocab_size and num_types must be provided for curve embedding")
            self.value_embedding = CurveEmbedding(num_types=num_types, vocab_size=vocab_size, d_model=d_model)
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            # Temporal embedding might not be relevant for curve sequence if it's not strictly time-aligned in the same way, 
            # or if we don't have time features for curves. 
            # But the model signature expects x_mark. If x_mark is provided, we use it.
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type='fixed', freq=freq) if freq!='t' else TimeFeatureEmbedding(d_model=d_model, embed_type='timeF', freq=freq)
        else:
            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.embed_type == 'curve':
            # For curve embedding, x is [Batch, SeqLen, 5] (LongTensor)
            # x_mark might be None or we might want to use it if we have timestamps for curves.
            # Assuming x_mark is provided and aligns with curve sequence.
            x = self.value_embedding(x) + self.position_embedding(x)
            if x_mark is not None:
                x = x + self.temporal_embedding(x_mark)
        else:
            x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)