import torch
import torch.nn.functional as F

def randomized_svd_torch(m, n, k, matvec=None, rmatvec=None, A_tensor=None,
                         p=10, n_iter=2, device='cpu', dtype=torch.float64,
                         random_seed=None):
    """
    Randomized truncated SVD for an m x n matrix A.
    Either provide A_tensor (torch.Tensor m x n) OR both matvec and rmatvec:
      - matvec(X): returns A @ X   (X shape n x l) -> returns m x l
      - rmatvec(Y): returns A.T @ Y (Y shape m x l) -> returns n x l

    Returns:
      U_k (m x k), S_k (k,), Vt_k (k x n)
    """
    if A_tensor is not None:
        A = A_tensor.to(device=device, dtype=dtype)
        def matvec_local(X):
            return A @ X
        def rmatvec_local(Y):
            return A.mH @ Y
        matvec = matvec_local
        rmatvec = rmatvec_local

    if matvec is None or rmatvec is None:
        raise ValueError("Provide either A_tensor or both matvec and rmatvec.")

    torch.manual_seed(random_seed if random_seed is not None else 0)

    l = k + p
    # 1) Draw random Gaussian test matrix Omega (n x l)
    Omega = torch.randn(n, l, device=device, dtype=dtype)

    # 2) Form Y = A @ Omega (m x l)
    Y = matvec(Omega)

    # 3) Power iterations (optional)
    for _ in range(n_iter):
        Z = rmatvec(Y)   # n x l
        Y = matvec(Z)    # m x l

    # 4) Orthonormalize Y -> Q (m x l)
    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # 5) Form small B = Q^T A (l x n). Compute AtQ = A^T @ Q -> n x l, then transpose.
    AtQ = rmatvec(Q)   # n x l
    B = AtQ.mH          # l x n

    # 6) Small SVD of B (cheap)
    U_B, S_all, Vt_all = torch.linalg.svd(B, full_matrices=False)

    # 7) Take top-k
    U_B_k = U_B[:, :k]      # l x k
    S_k = S_all[:k]         # k
    Vt_k = Vt_all[:k, :]    # k x n

    # 8) Expand left singular vectors: U = Q @ U_B_k (m x k)
    U_k = Q @ U_B_k

    return U_k, S_k, Vt_k


m = 20000
n = 20000
k = 8

A = torch.zeros((m, n), dtype=torch.complex64)
for i in range(10):
    a = torch.randn(n // 40, dtype=torch.complex64)
    ar = F.interpolate(a.real.unsqueeze(0).unsqueeze(0), size=n, mode='linear', align_corners=True).squeeze()
    ai = F.interpolate(a.imag.unsqueeze(0).unsqueeze(0), size=n, mode='linear', align_corners=True).squeeze()
    a = ar + 1j * ai
    u = torch.randn(m, dtype=torch.complex64)
    t = torch.linspace(0, 1, steps=m, dtype=torch.complex64)
    for j in range(100):
        u += (1/(1.2**j))*torch.exp(torch.rand(1, dtype=torch.complex64) * t)
    A += (1 / (1.5**i))*u.unsqueeze(1) * a.unsqueeze(0)


U, S, Vh = randomized_svd_torch(m, n, k, A_tensor=A, p = 15, n_iter = 3, device='cpu', dtype=torch.complex64)

A_reconstructed = (U * S.unsqueeze(0)) @ Vh

print('Hello: ', torch.linalg.norm(A - A_reconstructed) / torch.linalg.norm(A))