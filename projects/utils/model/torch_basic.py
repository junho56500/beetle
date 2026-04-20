import torch

torch.permute()
torch.flatten()
torch.reshape()
torch.view()
torch.contiguous().view()
torch.squeeze()
torch.unsqueeze()
torch.transpose()
torch.expand()
torch.repeat()
torch.split()
torch.chunk()
torch.cat()
torch.stack()

torch.narrow()
torch.gather()
torch.unbind()


torch.reshape(input, shape)
torch.view(shape)
torch.contiguous().view()
torch.unsqueeze(input, dim)      #(3, 224, 224) to a batch of one (1, 3, 224, 224)  
torch.squeeze(input, dim=None)   #(Batch, 1) when you just want (Batch).
torch.permute(*dims)            #tensor.permute(0, 2, 3, 1) moves channels from index 1 to index 3.
torch.transpose(input, dim0, dim1)      #swaps exactly two dimensions.
torch.flatten(input, start_dim=0, end_dim=-1)       #(Batch, C, H, W) $\rightarrow$ (Batch, C*H*W).
torch.expand(*sizes)
torch.repeat(*sizes)
torch.chunk(input, chunks, dim=0)
torch.split(tensor, split_size_or_sections, dim=0)
torch.cat(tensors, dim=0)       #If you cat them on the first dimension, you get a $(6, 224, 224)$ tensor.
torch.stack(tensors, dim=0)     # If you stack two $(3, 224, 224)$ images, you get a $(2, 3, 224, 224)$ tensor.

torch.narrow(input, dim, start, length)     #It is essentially a functional way of doing tensor[:, start:start+length].
torch.gather(input, dim, index)
torch.unbind(input, dim=0)      

