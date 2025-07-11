import torch


# Parameters:
# condition (Tensor): A boolean tensor. Its shape must be broadcastable with input and other.
# input (Tensor): The tensor whose elements are chosen when condition is True.
# other (Tensor): The tensor whose elements are chosen when condition is False.

# Return Value:
# A new tensor with elements chosen from input or other. The shape of the output tensor will be the broadcasted shape of condition, input, and other.

# --- Example 1: Basic 1D Tensors ---
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([1, 5, 3, 0])

result1 = torch.eq(a, b)
# Or using the operator:
# result1 = (a == b)

print(f"Example 1: Basic 1D Comparison")
print(f"Tensor a: {a}")
print(f"Tensor b: {b}")
print(f"a == b (torch.eq): {result1}")
# Expected Output: tensor([ True, False,  True, False])

print("\n" + "="*40 + "\n")

# --- Example 2: 2D Tensors ---
c = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
d = torch.tensor([[1, 0, 3],
                  [7, 5, 6]])

result2 = torch.eq(c, d)
print(f"Example 2: 2D Tensors")
print(f"Tensor c:\n{c}")
print(f"Tensor d:\n{d}")
print(f"c == d (torch.eq):\n{result2}")
# Expected Output:
# tensor([[ True, False,  True],
#         [False,  True,  True]])

print("\n" + "="*40 + "\n")

# --- Example 3: Comparing a Tensor with a Scalar ---
e = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])
scalar_val = 20

result3 = torch.eq(e, scalar_val)
print(f"Example 3: Tensor vs Scalar")
print(f"Tensor e:\n{e}")
print(f"Scalar: {scalar_val}")
print(f"e == scalar_val (torch.eq):\n{result3}")
# Expected Output:
# tensor([[False,  True, False],
#         [False, False, False]])

print("\n" + "="*40 + "\n")

# --- Example 4: Broadcasting ---
# f (shape 2,3) and g (shape 1,3)
f = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
g = torch.tensor([1, 5, 6]) # Shape (3,) - will be broadcast to (1,3) and then (2,3)

result4 = torch.eq(f, g)
print(f"Example 4: Broadcasting")
print(f"Tensor f:\n{f}")
print(f"Tensor g: {g}")
print(f"f == g (torch.eq):\n{result4}")
# Expected Output:
# tensor([[ True, False, False],  # (1,2,3) == (1,5,6) -> F F F -> should be True False False
#         [False, True,  True]]) # (4,5,6) == (1,5,6) -> F True True
# Correction:
# Tensor f: [[1, 2, 3], [4, 5, 6]]
# Tensor g: [1, 5, 6]
#
# f[0] == g -> [1==1, 2==5, 3==6] -> [T, F, F]
# f[1] == g -> [4==1, 5==5, 6==6] -> [F, T, T]
#
# So the output is correct based on broadcasting.

print("\n" + "="*40 + "\n")

# --- Example 5: Using with torch.where() ---
# A common pattern is to use torch.eq() (or ==) to create the condition mask for torch.where()
data = torch.tensor([10, 20, 30, 20, 40])
value_to_replace = 20
replacement = 0

mask_for_where = torch.eq(data, value_to_replace) # Create boolean mask
result_where = torch.where(mask_for_where, torch.tensor(replacement), data)
print(f"Example 5: Using with torch.where()")
print(f"Original data: {data}")
print(f"Value to replace ({value_to_replace}) mask: {mask_for_where}")
print(f"Result (20 replaced with 0): {result_where}")
# Expected Output: tensor([10, 0, 30, 0, 40])