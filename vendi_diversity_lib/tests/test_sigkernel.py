import torch
import sigkernel

# Specify the static kernel (for linear kernel use sigkernel.LinearKernel())
static_kernel = sigkernel.RBFKernel(sigma=0.5)

# Specify dyadic order for PDE solver (int > 0, default 0, the higher the more accurate but slower)
dyadic_order = 1

# Specify maximum batch size of computation; if memory is a concern try reducing max_batch, default=100
max_batch = 1

# Initialize the corresponding signature kernel
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

# Synthetic data
batch, len_x, len_y, dim = 250, 250, 70, 200
X = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda') # shape (batch,len_x,dim)
Y = torch.rand((batch,len_y,dim), dtype=torch.float64, device='cuda') # shape (batch,len_y,dim)
Z = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda') # shape (batch,len_y,dim)

# Compute signature kernel Gram matrix (i.e. k(x_i,y_j) for i,j=1,...,batch), also works for different batch_x != batch_y)
G = signature_kernel.compute_Gram(X,Y,sym=False,max_batch=max_batch)
print(G.shape)

print(G)