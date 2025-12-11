import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss that applies an ignore mask to exclude certain regions
    from contributing to the loss calculation.
    """
    def __init__(self,
                 reduction: str = 'mean', # 'mean' (over non-ignored pixels) or 'sum' (over non-ignored pixels)
                 ignore_index: int = -100 # Standard ignore index in PyTorch CrossEntropyLoss
                ):
        super().__init__()
        self.reduction_type = reduction
        self.ignore_index = ignore_index

        # We'll use CrossEntropyLoss with reduction='none' initially to get per-pixel loss
        self.ce_loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self,
                pred_logits: torch.Tensor, # (N, C, H, W) - Raw logits from model
                target_labels: torch.Tensor, # (N, H, W) - Ground truth class labels (long type)
                ignore_mask: torch.Tensor = None # (N, H, W) - Boolean or float (0/1) mask
                                                  # True/1 to KEEP, False/0 to IGNORE
                ) -> torch.Tensor:
        """
        Calculates Cross-Entropy Loss with an optional ignore mask.

        Args:
            pred_logits: Predicted class logits (before softmax/sigmoid).
                         Shape: (Batch_Size, Num_Classes, Height, Width).
            target_labels: Ground truth class labels (integer indices).
                           Shape: (Batch_Size, Height, Width).
            ignore_mask: Optional mask. Pixels where mask == 0 (or False) are ignored.
                         Shape: (Batch_Size, Height, Width).

        Returns:
            torch.Tensor: The calculated masked loss.
        """
        
        # 1. Calculate the element-wise Cross-Entropy Loss
        # This gives a loss value for each pixel (except those already at ignore_index)
        # Shape: (Batch_Size, Height, Width)
        pixel_wise_loss = self.ce_loss_func(pred_logits, target_labels)

        # 2. Apply the ignore mask
        if ignore_mask is not None:
            # Ensure mask is float type for multiplication and on the same device
            mask = ignore_mask.float().to(pixel_wise_loss.device)

            # Ensure mask is compatible with loss shape (B, H, W)
            if mask.shape != pixel_wise_loss.shape:
                raise ValueError(f"Ignore mask shape {mask.shape} does not match pixel-wise loss shape {pixel_wise_loss.shape}.")

            # Mask out (zero out) the loss for ignored regions
            masked_loss = pixel_wise_loss * mask
            
            # 3. Handle Reduction
            if self.reduction_type == 'mean':
                # Sum loss over all pixels, then divide by the *number of non-ignored pixels*
                # (sum of the mask itself, effectively counting True/1s)
                num_non_ignored_pixels = mask.sum()
                if num_non_ignored_pixels > 0:
                    loss = masked_loss.sum() / num_non_ignored_pixels
                else:
                    # If all pixels are ignored, return 0.0 loss to avoid division by zero
                    loss = torch.tensor(0.0, device=pred_logits.device)
            elif self.reduction_type == 'sum':
                # Sum loss over all non-ignored pixels
                loss = masked_loss.sum()
            else: # Should not happen due to constructor check
                raise ValueError("Unsupported reduction type.")
        else:
            # If no ignore_mask is provided, just apply standard reduction to pixel-wise loss
            if self.reduction_type == 'mean':
                loss = pixel_wise_loss.mean()
            elif self.reduction_type == 'sum':
                loss = pixel_wise_loss.sum()
            else:
                raise ValueError("Unsupported reduction type.")

        return loss

# --- Example Usage ---
if __name__ == '__main__':
    batch_size = 2
    num_classes = 5
    img_height = 4
    img_width = 4

    # Dummy model output (logits for each pixel)
    # (N, C, H, W)
    dummy_pred_logits = torch.randn(batch_size, num_classes, img_height, img_width)

    # Dummy ground truth labels (integer class IDs for each pixel)
    # (N, H, W)
    dummy_target_labels = torch.randint(0, num_classes, (batch_size, img_height, img_width), dtype=torch.long)

    # Dummy ignore mask (True/1 to keep, False/0 to ignore)
    # (N, H, W)
    # Let's ignore a few patches in the image, and also some random pixels
    dummy_ignore_mask = torch.ones(batch_size, img_height, img_width, dtype=torch.bool)
    
    # Ignore top-left 2x2 patch in first image of batch
    dummy_ignore_mask[0, 0:2, 0:2] = False
    # Ignore bottom-right 1x1 pixel in second image of batch
    dummy_ignore_mask[1, img_height-1, img_width-1] = False
    
    print("Dummy Ground Truth Labels (Batch 0):\n", dummy_target_labels[0])
    print("\nDummy Ignore Mask (Batch 0):\n", dummy_ignore_mask[0].int()) # print as int for clarity

    # --- Initialize the Masked Loss ---
    masked_loss_fn_mean = MaskedCrossEntropyLoss(reduction='mean')
    masked_loss_fn_sum = MaskedCrossEntropyLoss(reduction='sum')

    # --- Calculate Loss with Mask ---
    loss_mean = masked_loss_fn_mean(dummy_pred_logits, dummy_target_labels, dummy_ignore_mask)
    loss_sum = masked_loss_fn_sum(dummy_pred_logits, dummy_target_labels, dummy_ignore_mask)

    print(f"\nCalculated Masked Cross-Entropy Loss (mean): {loss_mean.item():.4f}")
    print(f"Calculated Masked Cross-Entropy Loss (sum): {loss_sum.item():.4f}")

    # --- Test Case: No mask (should be same as regular CE) ---
    print("\n--- Test Case: No Mask ---")
    regular_ce_loss = F.cross_entropy(dummy_pred_logits, dummy_target_labels, reduction='mean')
    loss_no_mask = masked_loss_fn_mean(dummy_pred_logits, dummy_target_labels, ignore_mask=None)
    print(f"Regular Cross-Entropy Loss: {regular_ce_loss.item():.4f}")
    print(f"Masked Loss (no mask): {loss_no_mask.item():.4f}")
    print(f"Are they close? {torch.isclose(regular_ce_loss, loss_no_mask)}")

    # --- Test Case: All pixels ignored ---
    print("\n--- Test Case: All Pixels Ignored ---")
    all_ignored_mask = torch.zeros_like(dummy_ignore_mask, dtype=torch.bool)
    loss_all_ignored = masked_loss_fn_mean(dummy_pred_logits, dummy_target_labels, all_ignored_mask)
    print(f"Loss when all pixels are ignored: {loss_all_ignored.item():.4f}") # Should be 0.0