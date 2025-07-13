import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Scenario: Multi-class classification logits ---
# Batch size = 2
# Number of classes = 3
# Each row represents the logits for a single sample across 3 classes.
logits = torch.tensor([
    [2.0, 1.0, 0.1],  # Logits for sample 1
    [0.5, 3.0, 1.5]   # Logits for sample 2
], dtype=torch.float32)

print("Original Logits:")
print(logits)
print("-" * 30)

# --- Using torch.nn.Softmax (as a module) ---
print("--- Using torch.nn.Softmax (Module) ---")
# Instantiate the Softmax module. We want probabilities for each sample,
# so the sum should be 1 across the class dimension (dim=1).
softmax_layer = nn.Softmax(dim=1)
probabilities_module = softmax_layer(logits)

print("Probabilities (nn.Softmax):\n", probabilities_module)
# Verify that each row sums to 1
print("Sum of probabilities for sample 1:", probabilities_module[0].sum().item())
print("Sum of probabilities for sample 2:", probabilities_module[1].sum().item())
print("-" * 30)

# --- Using torch.nn.functional.softmax (functional) ---
print("--- Using torch.nn.functional.softmax (Functional) ---")
# Directly call the functional version
probabilities_functional = F.softmax(logits, dim=1)

print("Probabilities (F.softmax):\n", probabilities_functional)
# Verify that each row sums to 1
print("Sum of probabilities for sample 1:", probabilities_functional[0].sum().item())
print("Sum of probabilities for sample 2:", probabilities_functional[1].sum().item())
print("-" * 30)

# --- What happens if you choose the wrong `dim`? ---
print("--- Choosing the Wrong `dim` ---")
# If we mistakenly apply softmax along dim=0 (columns)
softmax_wrong_dim = F.softmax(logits, dim=0)

print("Probabilities (F.softmax, dim=0 - WRONG for this case):\n", softmax_wrong_dim)
print("Sum of probabilities for column 0:", softmax_wrong_dim[:, 0].sum().item())
# This would be incorrect if each row represents a separate sample's class probabilities.
# The sums now appear vertically, which is usually not what you want for classification outputs.
print("-" * 30)

# --- Example with a 3D tensor (e.g., per-pixel classification logits) ---
print("--- Example with 3D Logits (e.g., Segmentation) ---")
# (Batch_Size, Num_Classes, Height, Width)
logits_3d = torch.randn(1, 4, 2, 2) # 1 image, 4 classes, 2x2 pixels
print("Original 3D Logits shape:", logits_3d.shape)

# For per-pixel classification, the softmax should be applied along the class dimension (dim=1)
# for each pixel independently.
softmax_3d = F.softmax(logits_3d, dim=1)
print("3D Probabilities shape (F.softmax, dim=1):", softmax_3d.shape)
print("Example pixel probabilities (first image, pixel 0,0):", softmax_3d[0, :, 0, 0].sum().item())
print("Example pixel probabilities (first image, pixel 1,1):", softmax_3d[0, :, 1, 1].sum().item())
# Here, each slice [:, h, w] sums to 1.
print("-" * 30)