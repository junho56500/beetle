import torch
import torch.nn as nn
import torch.nn.functional as F


#1. input q(leanable), v(bev, fp, lvl), reference points(static value)
#2. make learn for offset and weight from q
#3. add offset to referece points -> samples of grid
#4. final sample from v for grid
#5. weighted sum for samples and weights
#6. repeat for each level and add each output to final output



class MultiscaleDeformableAttention(nn.module):
    def __init__(self, d_model=256, n_lvl=4, n_heads=8, n_pts=4):
        super().__init__()
        self.d_model = d_model
        self.n_lvl = n_lvl
        self.n_heads = n_heads
        self.n_pts = n_pts
        self.d_head = d_model//n_heads
        
        self.sampling_offsets = nn.Linear(d_model, n_heads*n_lvl*n_pts*2)
        self.attn_weights = nn.Linear(d_model, n_heads*n_lvl*n_pts)
        self.value_proj = nn.Linear(d_model,d_model)
        self.output_proj = nn.Linear(d_model, d_model)
            
    def forward(self, q, reference_pts, feat_lvl):
        """
        Args:
            query: [B, Lq, C] (Object queries)
            reference_points: [B, Lq, 2] (Normalized 0-1 centers)
            feat_levels: List of [B, C, H_i, W_i] (Feature Pyramid)
        """
        b, lq, c = q.shape
        
        n_heads = self.n_heads
        n_pts = self.n_pts
        n_lvl = self.n_lvl
        
        #[b, lq, heads, lvl, pts, 2]
        offsets = self.sampling_offsets(q).view(b, lq, n_heads, n_lvl, n_pts, 2)
        #[b, lq, heads, lvl, pts]
        weights = self.attn_weights(q).view(b, lq, n_heads, n_lvl, n_pts)
        #[b, lq, heads, lvl, pts]
        weights = F.softmax(weights, dim=-1).view(b, lq, n_heads, n_lvl,  n_pts)
        #[b, lq, c]
        output = torch.zeros(b, lq, c, device=q.device)
        
        for lvl, x in enumerate(feat_lvl):
            # x_v: [B, head_dim * n_heads, H, W]  c = head_dim*n_heads? -> [b,c,h,w]
            # c=256 n_head=8 d_head=32
            x_v = self.value_proj(x.permutate(0, 2, 3, 1)).permute(0, 3, 1, 2)
            #-> [B * n_heads, head_dim, H, W]
            x_v = x_v.view(b * n_heads, self.d_head, x.shape[2], x.shape[3])
            lvl_offsets = offsets[:,:,:,lvl,:,:] #[b, lq, n_heads, n_pts, 2]
            
            #[b,lq,n_heads,n_pts,2]
            ref_lvl = reference_pts.view(b, lq, 1, 1, 2)
            sampling_locs = ref_lvl + lvl_offsets
            
            #[b*n_heads, lq, n_pts, 2]
            locs_for_grid = sampling_locs.transpose(1, 2).flatten(0, 1) # [B*n_heads, Lq, n_points, 2]
            # Rescale 0~1 to -1~1 for PyTorch grid_sample
            locs_for_grid = locs_for_grid * 2 - 1
            
            #[b*n_head, d_head, lq, n_pts]
            sampled = F.grid_sample(x_v, locs_for_grid, align_corners=False)
            
            # weights: [B, Lq, n_heads, n_levels, n_points]
            # lvl_weights: [B * n_heads, 1, Lq, n_points]
            lvl_weights = weights[:, :, :, lvl, :].transpose(1, 2).reshape(B * n_heads, 1, lq, n_pts)
            
            #[b*heads,d_head,lq]
            weight_sampled = (sampled * lvl_weights).sum(-1)
            
            #[b,lq,heads,d_head] -> #[b,lq,heads*d_head]``
            output += weight_sampled.view(b, n_heads, self.head_dim, lq).permute(0, 3, 1, 2).flatten(2)
            
        return self.output_proj(output)
    
    
b, lq, c = 4, 20, 256
feat_lvl = [torch.randn(b, c, 64//(2**i), 64//(2*i)) for i in range(4)]  #[b,c,h,w]
query = torch.randn(b, lq, c)
ref_pts = torch.randn(b, lq, 2)

model = MultiscaleDeformableAttention(d_model=c)
out = model(query, ref_pts, feat_lvl)
