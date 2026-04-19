# 1. Linear Layers (The Prediction Heads)
# For Classification (Labeling the object)
# Activation: * Hidden Layers: ReLU or GeLU.

# Final Layer: Softmax (if classes are mutually exclusive) or 
# Sigmoid (if it's multi-label classification).

# Loss Function: Cross-Entropy Loss (nn.CrossEntropyLoss). This measures the distance between 
# the predicted probability distribution and the ground truth (one-hot vector).

# For Position/Bounding Boxes (Coordinates)
# Activation: * Final Layer: Sigmoid (if coordinates are normalized between 0 and 1) 
# or None/Identity (if predicting raw pixel offsets).

# Loss Function: * L1 Loss (Mean Absolute Error): Robust to outliers.
# Smooth L1 Loss: A hybrid that is steady like L1 but smooth near zero like L2.
# GIoU/DIoU Loss: Specialized for boxes to account for overlapping areas.


# 2. CNNs (The Feature Extractor)
# CNNs serve as the "eyes" of the model, extracting spatial hierarchies.
# Activation: ReLU is the standard. In more modern architectures, 
# SiLU (Swish) is used (e.g., in YOLOv8) because it is smoother and helps gradients 
# flow better during deep backpropagation.

# Loss Function: CNNs aren't usually trained with a standalone loss; 
# they receive the "gradient signal" flowing back from the Classification and Position heads.

# 3. Attention (The Context Provider)
# Attention helps the model understand the relationship between different parts of an image
# (e.g., "This tail belongs to that cat body").

# Activation: * Internal: Softmax is used 
# inside the attention mechanism to weight the importance of different pixels.

# Feed-Forward Blocks: GeLU (Gaussian Error Linear Unit) is the industry standard for Transformers/Attention layers because it handles negative values more gracefully than ReLU.

# Loss Function: In models like DETR (Detection Transformer), a Bipartite Matching Loss 
# (Hungarian Loss) is used to unique match predicted boxes to ground truth boxes.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=256):
        super().__init__()
        
        # 1. CNN Backbone (Feature Extraction)
        # Usage: ReLU activation, Kaiming Init
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        
        # 2. Attention Layer (Context)
        # Usage: GeLU activation, Xavier Init
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 3. Classification Head (Labels)
        # Usage: Softmax (via CrossEntropy), Xavier Init
        self.class_head = nn.Linear(embed_dim, num_classes)
        
        # 4. Regression Head (Position: x, y, w, h)
        # Usage: Sigmoid (for 0-1 coords), Zero/Small Constant Init
        self.bbox_head = nn.Linear(embed_dim, 4)

        # Apply custom initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            # Use Xavier for the class head
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)
            nn.init.xavier_uniform_(m.out_proj.weight)

    def forward(self, x):
        # CNN Pass
        features = self.backbone(x) # [Batch, 256, 7, 7]
        features = features.flatten(2).permute(0, 2, 1) # [Batch, 49, 256]
        
        # Attention Pass
        attn_out, _ = self.attention(features, features, features)
        features = self.layer_norm(features + attn_out)
        
        # Pull global representation (mean across spatial locations)
        global_feat = features.mean(dim=1)
        
        # Output Heads
        class_logits = self.class_head(global_feat)
        bbox_coords = torch.sigmoid(self.bbox_head(global_feat)) # Sigmoid constrains to [0, 1]
        
        return class_logits, bbox_coords

# --- Training Setup (Loss Functions) ---

num_classes = 10
model = DetectionTransformer(num_classes=num_classes)

# Classification Loss: Handles Softmax internally
criterion_cls = nn.CrossEntropyLoss()

# Position Loss: SmoothL1 is less sensitive to outliers than MSE
criterion_bbox = nn.SmoothL1Loss()