import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define our dummy data ---
batch_size = 2
sequence_length = 5
num_classes = 10
padding_idx = 0 # Let's assume class 0 is our padding token

# Simulate model outputs (logits)
# shape: (batch_size, sequence_length, num_classes)
# In a real model, this would come from a linear layer output
logits = torch.randn(batch_size, sequence_length, num_classes)

# Simulate true target labels
# shape: (batch_size, sequence_length)
# Note: For CrossEntropyLoss, targets are just class indices
targets = torch.tensor([
    [1, 2, 3, 0, 0],  # Sample 1: Actual sequence 1,2,3, then padding
    [4, 5, 0, 0, 0]   # Sample 2: Actual sequence 4,5, then padding
])

print("--- Initial Data ---")
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
print("Targets (with padding):")
print(targets)
print("-" * 30)

# --- 2. Calculate Unreduced Loss (reduction='none') ---

# CrossEntropyLoss expects logits in shape (N, C) and targets in shape (N)
# where N is the total number of items to classify.
# So, we need to reshape our data.

# Reshape logits: (batch_size * sequence_length, num_classes)
logits_flat = logits.view(-1, num_classes)
# Reshape targets: (batch_size * sequence_length)
targets_flat = targets.view(-1)

# Calculate the loss for each individual element
# This returns a tensor of shape (batch_size * sequence_length)
per_element_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

print("--- Per-Element Loss (reduction='none') ---")
print("Shape:", per_element_loss.shape)
print("Values:", per_element_loss)
print("-" * 30)

# --- 3. Masking: Ignore Padding Tokens ---

# Create a mask to identify non-padding tokens
# Where targets are NOT padding_idx, the mask is True (1.0), otherwise False (0.0)
mask = (targets_flat != padding_idx).float()

print("--- Mask for Padding ---")
print("Mask shape (flat):", mask.shape)
print("Mask values (flat):", mask)
print("-" * 30)

# Apply the mask to the per-element loss
# This will set the loss for padding tokens to 0
masked_loss = per_element_loss * mask

print("--- Masked Loss ---")
print("Masked loss values:", masked_loss)
print("-" * 30)

# --- 4. Weighting: Assign custom weights to specific tokens/classes ---

# Let's say we want to give more importance to class 2 and class 5
# and less importance to class 1.
# This could also be a per-sample weight if you have a reason for that.

# For simplicity, let's create a per-token weight based on the target class.
# This would typically be a tensor of weights, where index corresponds to class ID.
# Let's say: class 1 gets weight 0.5, class 2 gets 2.0, class 5 gets 1.5, others 1.0
class_weights_map = {
    1: 0.5,
    2: 2.0,
    5: 1.5
}

# Create a weight tensor for each element based on its target class
# Initialize all weights to 1.0
weights = torch.ones_like(targets_flat, dtype=torch.float)

for class_id, weight_value in class_weights_map.items():
    weights[targets_flat == class_id] = weight_value

print("--- Custom Weights per Element ---")
print("Weights shape:", weights.shape)
print("Weights values:", weights)
print("-" * 30)

# Apply the custom weights to the masked loss
weighted_masked_loss = masked_loss * weights

print("--- Weighted Masked Loss ---")
print("Weighted masked loss values:", weighted_masked_loss)
print("-" * 30)

# --- 5. Final Aggregation for Backward Pass ---

# To get a scalar value for backward(), we usually sum the weighted and masked losses
# and then divide by the total number of *actual* (non-padded) elements.
# This gives us an average loss over the meaningful tokens.

# Sum of the mask gives us the count of non-padded tokens
num_non_padded_tokens = mask.sum()

# If num_non_padded_tokens is 0 (e.g., an empty batch or all padding), avoid division by zero
final_loss = weighted_masked_loss.sum() / num_non_padded_tokens \
             if num_non_padded_tokens > 0 else torch.tensor(0.0)


print("--- Final Scalar Loss for Backward Pass ---")
print("Number of non-padded tokens:", num_non_padded_tokens.item())
print("Final Loss (scalar):", final_loss.item())
print("-" * 30)

# --- 6. Backward Pass (demonstration) ---

# In a real training loop, you would typically:
# 1. Zero out gradients: optimizer.zero_grad()
# 2. Compute gradients: final_loss.backward()
# 3. Update parameters: optimizer.step()

# Simulate backward pass (requires parameters to optimize)
# Let's pretend logits came from a simple linear layer
linear_layer = nn.Linear(num_classes, num_classes) # Dummy layer
# Re-create logits with requires_grad=True
logits = linear_layer(torch.randn(batch_size, sequence_length, num_classes, requires_grad=True))
logits_flat = logits.view(-1, num_classes)

# Recalculate everything up to final_loss
per_element_loss_b = F.cross_entropy(logits_flat, targets_flat, reduction='none')
masked_loss_b = per_element_loss_b * mask
weighted_masked_loss_b = masked_loss_b * weights
num_non_padded_tokens_b = mask.sum()
final_loss_b = weighted_masked_loss_b.sum() / num_non_padded_tokens_b \
               if num_non_padded_tokens_b > 0 else torch.tensor(0.0)


print("--- Demonstrating Backward Pass ---")
print("Final loss (for backward):", final_loss_b.item())

# Perform the backward pass
final_loss_b.backward()

# Now, gradients would be computed for 'logits' and any parameters that led to it.
# For example, if 'logits' were the output of 'linear_layer',
# you could inspect linear_layer.weight.grad
print("Gradients for dummy logits:", logits.grad)
print("-" * 30)