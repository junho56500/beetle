import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class EmbeddingHierarchicalQuery(nn.Module):
    def __init__(self, embed_dims=256, num_points=20, num_instances=50):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.num_instances = num_instances
        self.attention = nn.MultiheadAttention(embed_dims, 8, batch_first=True)

    def forward(self, query, key, value):
        # query shape: [batch, num_instances * num_points, dims]
        B, Total_Q, C = query.shape
        
        # --- 1. Intra-instance Attention ---
        # Reshape to group points of the same instance together
        # New shape: [batch * num_instances, num_points, dims]
        intra_query = query.view(B * self.num_instances, self.num_points, C)
        
        # Points only look at other points in the same line
        intra_out, _ = self.attention(intra_query, intra_query, intra_query)
        intra_out = intra_out.view(B, Total_Q, C)

        # --- 2. Inter-instance Attention ---
        # Usually, we pool the points to get one "instance feature" 
        # or use a specific "instance token"
        # Reshape to: [batch * num_points, num_instances, dims]
        inter_query = query.view(B, self.num_instances, self.num_points, C)
        inter_query = inter_query.permute(0, 2, 1, 3).reshape(B * self.num_points, self.num_instances, C)
        
        # Instances look at each other at the same point index
        inter_out, _ = self.attention(inter_query, inter_query, inter_query)
        
        # Reshape back to original query format
        inter_out = inter_out.view(B, self.num_points, self.num_instances, C).permute(0, 2, 1, 3).reshape(B, Total_Q, C)

        return intra_out + inter_out
    
    
    
    import torch
import torch.nn as nn

class HierarchicalQueryGenerator(nn.Module):
    def __init__(self, embed_dims=256, num_points=20, num_instances=50):
        super().__init__()
        self.num_instances = num_instances
        self.num_points = num_points
        self.total_queries = num_instances * num_points
        
        # 1. Use nn.Embedding as the "container" for learnable queries
        # This creates a lookup table of shape [total_queries, embed_dims]
        self.query_embed = nn.Embedding(self.total_queries, embed_dims)
        
        # 2. Your hierarchical logic class
        self.hierarchical_layer = EmbeddingHierarchicalQuery(
            embed_dims=embed_dims, 
            num_points=num_points, 
            num_instances=num_instances
        )

    def forward(self, batch_size):
        # Create indices [0, 1, 2, ..., total_queries-1]
        device = self.query_embed.weight.device
        indices = torch.arange(self.total_queries, device=device)
        
        # Look up the learnable vectors
        # Shape: [total_queries, embed_dims]
        queries = self.query_embed(indices)
        
        # Expand for the batch
        # Shape: [batch, total_queries, embed_dims]
        queries = queries.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Pass through the hierarchical attention layers
        # (Using None for K/V as per your current class implementation)
        query_features = self.hierarchical_layer(queries, None, None)
        
        return query_features

# Usage
generator = HierarchicalQueryGenerator()
batch_out = generator(batch_size=8) # Returns [8, 1000, 256]