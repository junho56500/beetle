import torch

#torch.arange(start=0, end, step=1, *, dtype=None, layout=torch.strided, device=None, requires_grad=False)

# Values from 0 up to (but not including) 5
a = torch.arange(5)
print(a)
# Output: tensor([0, 1, 2, 3, 4])

# Values from 2 up to (but not including) 8
b = torch.arange(2, 8)
print(b)
# Output: tensor([2, 3, 4, 5, 6, 7])

# Values from 1 to 10, with a step of 2
c = torch.arange(1, 10, 2)
print(c)
# Output: tensor([1, 3, 5, 7, 9])

# Descending sequence with negative step
d = torch.arange(10, 0, -1)
print(d)
# Output: tensor([10,  9,  8,  7,  6,  5,  4,  3,  2,  1])

# Float steps
e = torch.arange(0.5, 3.5, 0.7)
print(e)
# Output: tensor([0.5000, 1.2000, 1.9000, 2.6000, 3.3000])

# Explicitly set data type to float32
f = torch.arange(0, 5, dtype=torch.float32)
print(f)
# Output: tensor([0., 1., 2., 3., 4.])
print(f.dtype)
# Output: torch.float32

# Inferred dtype (default is often int64 if all inputs are integers)
g = torch.arange(0, 5)
print(g.dtype)
# Output: torch.int64

# Create a tensor on GPU (if CUDA is available)
if torch.cuda.is_available():
    h = torch.arange(5, device='cuda')
    print(h)
    print(h.device)
    # Output: tensor([0, 1, 2, 3, 4], device='cuda:0')
    # Output: cuda:0
else:
    print("CUDA not available, creating on CPU.")
    h = torch.arange(5, device='cpu')
    print(h)
    print(h.device)
    # Output: tensor([0, 1, 2, 3, 4])
    # Output: cpu
    
# Create a tensor that tracks gradients
i = torch.arange(5.0, requires_grad=True)
print(i)
# Output: tensor([0., 1., 2., 3., 4.], requires_grad=True)