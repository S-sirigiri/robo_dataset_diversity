import numpy as np
import ksig

# Number of signature levels to use.
n_levels = 5

# Use 100 components in RFF and projection.
n_components = 100

# Instantiate RFF feature map.
static_feat = ksig.static.features.RandomFourierFeatures(n_components=n_components)
# Instantiate tensor random projections.
proj = ksig.projections.TensorizedRandomProjection(n_components=n_components)

# The RFSF-TRP feature map and kernel. Additionally to working as a callable for
# computing a kernel, it implements a fit and a transform method.
rfsf_trp_kernel = ksig.kernels.SignatureFeatures(
    n_levels=n_levels, static_features=static_feat, projection=proj)

# Generate 1000 sequences of length 200 with 100 features.
n_seq, l_seq, n_feat = 250, 200, 100
X = np.random.randn(n_seq, l_seq, n_feat)

# Fit the kernel to the data.
rfsf_trp_kernel.fit(X)

# Compute the kernel matrix as before.
K_XX = rfsf_trp_kernel(X)  # K_XX has shape (1000, 1000).

# Generate another array of 800 sequences of length 250 and 100 features.
n_seq2, l_seq2 = 800, 250
Y = np.random.randn(n_seq2, l_seq2, n_feat)

# Compute the kernel matrix between X and Y.
# The kernel does not have to be fitted a second time.
K_XY = rfsf_trp_kernel(X, Y)  # K_XY has shape (1000, 800)

# Alternatively, we may compute features separately for X and Y. Under the hood,
# this is what the call method does, i.e. compute features and take their inner product.
P_X = rfsf_trp_kernel.transform(X)  # P_X has shape (1000, 501)
P_Y = rfsf_trp_kernel.transform(Y)  # P_Y has shape (800, 501)

# Check that the results match.
print(np.linalg.norm(K_XX - P_X @ P_X.T))
print(np.linalg.norm(K_XY - P_X @ P_Y.T))