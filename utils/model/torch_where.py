import torch

# torch.where(condition, input, other, *, out=None) -> Tensor
# Parameters:
# condition (Tensor): A boolean tensor. Its shape must be broadcastable with input and other.
# input (Tensor): The tensor whose elements are chosen when condition is True.
# other (Tensor): The tensor whose elements are chosen when condition is False.

# Return Value:
# A new tensor with elements chosen from input or other. The shape of the output tensor will be the broadcasted shape of condition, input, and other.

# Example 1: Basic usage with 1D tensors
# Select from x if condition is True, else from y
cond1 = torch.tensor([True, False, True, False])
x1 = torch.tensor([10, 20, 30, 40])
y1 = torch.tensor([1, 2, 3, 4])

result1 = torch.where(cond1, x1, y1)
print(f"Example 1: Basic 1D Selection\nCondition: {cond1}\nInput (x): {x1}\nOther (y): {y1}\nResult: {result1}")
# Expected Output: [10, 2, 30, 4]

print("-" * 30)

# Example 2: Using broadcasting
# condition (1D) broadcasts to match x2 and y2 (2D)
cond2 = torch.tensor([True, False, True]) # Shape (3,)
x2 = torch.tensor([[1, 1, 1],
                   [2, 2, 2],
                   [3, 3, 3]]) # Shape (3, 3)
y2 = torch.tensor([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]) # Shape (3, 3)

# For row 0 (True): pick from x2's row 0
# For row 1 (False): pick from y2's row 1
# For row 2 (True): pick from x2's row 2
result2 = torch.where(cond2.unsqueeze(1), x2, y2) # Unsqueeze cond2 to (3,1) for row-wise broadcast
print(f"Example 2: Broadcasting Condition\nCondition: {cond2.unsqueeze(1)}\nInput (x):\n{x2}\nOther (y):\n{y2}\nResult:\n{result2}")
# Expected Output:
# tensor([[1, 1, 1],
#         [0, 0, 0],
#         [3, 3, 3]])

print("-" * 30)

# Example 3: Single scalar condition with tensors
# If condition is a single True, pick all from x3, else pick all from y3
cond3 = torch.tensor(True) # Scalar condition
x3 = torch.randn(2, 2)
y3 = torch.zeros(2, 2)

result3_true = torch.where(cond3, x3, y3)
print(f"Example 3: Scalar Condition (True)\nCondition: {cond3}\nInput (x):\n{x3}\nOther (y):\n{y3}\nResult:\n{result3_true}")

cond3_false = torch.tensor(False) # Scalar condition
result3_false = torch.where(cond3_false, x3, y3)
print(f"\nExample 3: Scalar Condition (False)\nCondition: {cond3_false}\nInput (x):\n{x3}\nOther (y):\n{y3}\nResult:\n{result3_false}")

print("-" * 30)

# Example 4: Implementing ReLU manually (for illustration)
# Where x is positive, keep x, else 0
x4 = torch.tensor([-1.0, 0.5, -2.0, 3.0])
relu_manual = torch.where(x4 > 0, x4, torch.tensor(0.0))
print(f"Example 4: Manual ReLU with torch.where\nInput: {x4}\nResult (ReLU): {relu_manual}")
# Expected Output: tensor([0.0, 0.5, 0.0, 3.0])

print("-" * 30)

# Example 5: Replacing values in a tensor
# Replace all values in data5 that are less than 5 with 0
data5 = torch.tensor([[1, 6, 3],
                      [8, 2, 7]])
threshold = 5
mask = data5 < threshold
result5 = torch.where(mask, torch.tensor(0), data5)
print(f"Example 5: Replacing values based on condition\nOriginal:\n{data5}\nThreshold: {threshold}\nMask:\n{mask}\nResult:\n{result5}")
# Expected Output:
# tensor([[0, 6, 0],
#         [8, 0, 7]])