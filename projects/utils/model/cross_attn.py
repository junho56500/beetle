import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(context_dim, embed_dim)
        self.val = nn.Linear(context_dim, embed_dim)
        
        self.scale = embed_dim**-0.5
        
    def forward(self, x, context):
        query = self.query(x)
        key = self.key(context)
        val = self.val(context)
        
        output = nn.softmax(query@key.transpose(-2, -1)*self.scale, dim=-1) @ val
        
        return output
    

b, seq_len, ctx_len = 1, 10, 5
dim, ctx_dim = 64, 128

x = torch.randn(b, seq_len, dim)
ctx = torch.randn(b, ctx_len, ctx_dim)

cross_attn = CrossAttention(dim, ctx_dim)
output = cross_attn(x, ctx)

#print(output)