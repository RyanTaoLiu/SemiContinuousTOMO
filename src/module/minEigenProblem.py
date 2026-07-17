import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def forward(A, B, x0, max_iter=50, tol=1e-8):
    dtype, device = A.dtype, A.device

    # --- Cholesky (fallback with jitter) ---
    try:
        L = torch.linalg.cholesky(A)
    except RuntimeError:
        jitter = 1e-4 * torch.eye(A.shape[0], dtype=dtype, device=device)
        L = torch.linalg.cholesky(A + jitter)

    # --- initialize vector ---
    v = x0.clone()

    # --- prepare projection matrix P: project to {v | v[1] = 0} ---
    n = A.shape[0]
    e1 = torch.zeros((n, 1), dtype=dtype, device=device)
    e1[1, 0] = 1.0
    P = torch.eye(n, dtype=dtype, device=device) - e1 @ e1.T

    for _ in range(max_iter):
        # Solve (A)x = Bv
        v = torch.cholesky_solve(B @ v, L, upper=False)

        # Project v to subspace v[1] = 0
        v = P @ v

        # Normalize under B-inner product
        v = v / torch.sqrt(v.T @ B @ v + tol)

    # Rayleigh quotient
    lam = (v.T @ A @ v) / (v.T @ B @ v)

    # Optional orthogonal direction construction
    '''
    v0 = v.clone()
    v2 = v0.reshape(-1, 2)
    vT = torch.stack([-v2[:, 1], v2[:, 0]], dim=1).reshape([-1, 1])
    beta = vT[0] / v0[0]
    v = v0 + beta * vT
    v = v / v.norm()
    print('v.norm', v.norm()) 
    '''
    return lam.squeeze(), v

class MinEigenProblem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, x0, max_iter=50, tol=1e-8):
        dtype, device = A.dtype, A.device

        # --- Cholesky for inverse iteration ---------------------------
        try:
            L = torch.linalg.cholesky(A)
        except RuntimeError:
            jit = 1e-4 * torch.eye(A.shape[0], dtype=dtype, device=device)
            L = torch.linalg.cholesky(A + jit)

        v = x0.clone()
        for _ in range(max_iter):
            v = torch.cholesky_solve(B @ v, L, upper=False)
            v = v / torch.sqrt(v.T @ B @ v + tol)

        lam = (v.T @ A @ v) / (v.T @ B @ v)
        v0 = v.clone()
        v2 = v0.reshape(-1, 2)
        vT = torch.stack([-v2[:, 1], v2[:, 0]], dim=1).reshape([-1,1])
        beta = vT[0] / v0[0]
        v = v0 + beta * vT
        v = v / v.norm()
        print('v.norm', v.norm()) 
        ctx.save_for_backward(A, B, lam, v)
        return lam.squeeze(), v

    # ------------------------------------------------------------
    @staticmethod
    def backward(ctx, dlam, dv):
        A, B, lam, v = ctx.saved_tensors
        dtype, device = A.dtype, A.device
        n = A.shape[0]

        # ---------- dλ/dA -----------------------------------------
        dlam_dA = (v @ v.T) / (v.T @ B @ v)
        A_grad  = dlam * dlam_dA

        # ---------- if dv!=0, need dΨ/dA ------------------------
        if dv is not None and dv.abs().sum().is_nonzero():
            Bv   = B @ v
            e_bk = torch.zeros_like(Bv); e_bk[1] = 1.0

            # -- KKT Matric H  (n+2) × (n+2) --
            H_top = torch.cat((A - lam * B, -Bv,     e_bk), dim=1)
            H_mid = torch.cat((-Bv.T,       torch.zeros(1,2, dtype=dtype, device=device)), dim=1)
            H_bot = torch.cat((e_bk.T,      torch.zeros(1,2, dtype=dtype, device=device)), dim=1)
            H     = torch.cat((H_top, H_mid, H_bot), dim=0)

            # -- pseudoinverse-base RHS (n+2)×n --
            rhs_basis = torch.cat(
                (torch.eye(n, dtype=dtype, device=device),
                 torch.zeros(2, n, dtype=dtype, device=device)),
                dim=0
            )

            # -- solve H X = rhs_basis  →  X:(n+2,n) --
            
            # X = torch.linalg.solve(H, rhs_basis)        
            H_np = H.detach().cpu().numpy()
            rhs_basis_np = rhs_basis.detach().cpu().numpy()
            X = np.linalg.lstsq(H_np, rhs_basis_np, rcond=None)[0]
            X = torch.from_numpy(X).to(dtype).to(device)
            X_PHI = X[:-2, :]
            dv_PHI = dv.T @ X_PHI
            A_grad += -0.5 * (torch.outer(dv_PHI.squeeze(), v.squeeze()) +
                              torch.outer(v.squeeze(), dv_PHI.squeeze()))
        return A_grad, None, None, None   # dB=0, dx0=0, dmax_iter=None

class MinEigenProblemSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, x0):
        # A: sparse tensor, x0: dense vector
        dtype, device = A.dtype, A.device
        coo = A.coalesce()  
        rows = coo.indices()[0].cpu().numpy()
        cols = coo.indices()[1].cpu().numpy()
        vals = coo.values().cpu().numpy()
        A_sp = sp.coo_matrix((vals, (rows, cols)), shape=coo.shape)
        lam, v0 = spla.eigsh(A_sp, k=1, return_eigenvectors=True, which='SA')
        v2 = v0.reshape(-1, 2)
        vT = np.stack([-v2[:, 1], v2[:, 0]], axis=1).reshape([-1,1])
        beta = vT[0] / v0[0]
        v = v0 + beta * vT
        v = v / np.linalg.norm(v)

        # ctx.save_for_backward(A, lam, v)
        ctx.A_sp = A_sp
        ctx.lam = lam
        ctx.v = v
        ctx.dtype = dtype
        ctx.device = device
        ctx.row = A_sp.row
        ctx.col = A_sp.col
        return torch.from_numpy(lam).to(dtype).to(device), torch.from_numpy(v).to(dtype).to(device)

    @staticmethod
    def backward(ctx, dlam, dv):
        A_sp, lam, v = ctx.A_sp, ctx.lam, ctx.v
        row = ctx.row
        col = ctx.col
        dtype, device = ctx.dtype, ctx.device
        n = A_sp.shape[0]

         # Constraint vector
        e_bk = np.zeros(n)
        e_bk[1] = 1.0

        # Top block (A - λI), shape: (n, n+2)
        I = sp.eye(n, format="csr")
        A_lamI = A_sp - lam[0] * I
        H_top = sp.hstack([A_lamI, -v.reshape(-1, 1), e_bk.reshape(-1, 1)])

        # Middle & bottom rows (1×(n+2))
        H_mid = sp.hstack([-v.reshape(1, -1), sp.csr_matrix((1, 2))])
        H_bot = sp.hstack([e_bk.reshape(1, -1), sp.csr_matrix((1, 2))])

        # Full KKT matrix: shape (n+2, n+2)
        H = sp.vstack([H_top, H_mid, H_bot]).tocsr()

        # RHS: shape (n+2, n)
        rhs = np.vstack([
            np.eye(n),
            np.zeros((2, n))
        ])
        rhs = np.concatenate((dv.detach().cpu().numpy(), np.zeros((2,1))))

        # Solve H x = rhs
        X = spla.spsolve(H, rhs)
        X_PHI = X[:-2, :]
        dvT = dv.T.detach().cpu().numpy()
        dv_PHI = (dvT @ X_PHI).squeeze()
        vf = v.squeeze()
        '''
        A_grad = -0.5 * (np.outer(dv_PHI.squeeze(), v.squeeze()) +
                            np.outer(v.squeeze(), dv_PHI.squeeze()))
        '''
        A_grad_value = 0.5 * (dv_PHI[row.squeeze()]*vf[col.squeeze()] + dv_PHI[col.squeeze()]*vf[row.squeeze()])
        A_grad = torch.sparse_coo_tensor((row, col), A_grad_value, A_sp.shape, dtype=dtype, requires_grad=True)
        return A_grad, None

# directly solve via inverse method
class inversePowerMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, x0):
        try:
            L = torch.linalg.cholesky(A)
        except:
            print('L is not SPD!')
            if torch.isnan(A).any():
                print('   Checked: A contains Nan!')
                raise ('L include Nan!')

        PsiList = []
        Psi = x0.clone()
        PsiList.append(Psi.clone())
        for _ in range(20):
            Psi = torch.cholesky_solve(B @ Psi, L)
            PsiList.append(Psi.clone())
            Psi = Psi / (torch.sqrt(Psi.T @ B @ Psi))
            PsiList.append(Psi.clone())

        ctx.save_for_backward(A, B, L)
        ctx.psiList = PsiList
        return Psi


    @staticmethod
    def backward(ctx, grad_psi):
        A, B, L = ctx.saved_tensors
        Psi = ctx.psiList

        A_grad = torch.zeros_like(A)
        iteration_times = (len(Psi) - 1) // 2
        grad_psi_2n = grad_psi
        for i in range(iteration_times):
            n = 2 * i
            xbx = Psi[-n - 2].T @ B @ Psi[-n - 2]
            # Psi_{2n} => Psi_{2n-1} => Psy_{2n} = Psy_{2n-1} / sqrt(Psy_{2n-1}.T @ B @ Psy_{2n-1})
            grad_Psi_2nm1 = xbx ** (-0.5) * grad_psi_2n - xbx ** (-1.5) * Psi[-n - 2] * torch.dot(
                (B @ Psi[-n - 2]).flatten(), grad_psi_2n.flatten())

            # Psi_{2n-1} => Psi_{2n} => Psy_{2n-1} = A^-1 @ B @ Psy_{2n-2}
            grad_Psi2nm1_2nm2 = torch.cholesky_solve(B @ grad_Psi_2nm1, L)

            # Psi_{2n-1} => A => Psy_{2n-1} = Psy_{2n-1} = A^-1 @ B @ Psy_{2n-2}
            grad_Psi2nm1_A_half = torch.outer(torch.cholesky_solve(grad_Psi_2nm1, L).flatten(), Psi[-n - 2].flatten())
            grad_Psi2nm1_A = -(grad_Psi2nm1_A_half + grad_Psi2nm1_A_half.T) / 2

            grad_psi_2n = grad_Psi2nm1_2nm2
            A_grad += grad_Psi2nm1_A
        return A_grad, None, None

class stripeProblem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edgeVal, row_np, col_np, max_iter=20, tol=1e-8):
        dtype, device = edgeVal.dtype, edgeVal.device
        edgeValNp = edgeVal.detach().cpu().numpy()
        angle = edgeValNp
        nedge = edgeValNp.shape[0]
        row = row_np.astype(np.int32)
        col = col_np.astype(np.int32)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        val = 0.5 * np.hstack([cos_angle,-sin_angle,sin_angle,cos_angle,cos_angle,-sin_angle,sin_angle,cos_angle]).flatten()
        A = sp.coo_matrix((val, (row, col)), shape=(nedge*2, nedge*2))
        lam, v0 = spla.eigsh(A, k=1, return_eigenvectors=True, which='SA')
        v2 = v0.reshape(-1, 2)
        vT = np.stack([-v2[:, 1], v2[:, 0]], axis=1).reshape(-1)
        beta = vT[0] / v0[0]
        v = v0 + beta * vT
        v = v / v.norm()
        # v = alpha * v0 + beta * vT
        lam_th = torch.from_numpy(lam).to(dtype).to(device)
        v_th = torch.from_numpy(v).to(dtype).to(device)

        # ctx.save_for_backward(angle, row_np, col_np, lam, v)
        ctx.angle = angle
        ctx.row_np = row_np
        ctx.col_np = col_np
        ctx.lam = lam
        ctx.v = v
        ctx.device = device
        return lam_th, v_th

    @staticmethod
    def backward(ctx, dlam, dv):
        # angle, row_np, col_np, lam, v, device = ctx.saved_tensors
        angle = ctx.angle
        row_np = ctx.row_np
        col_np = ctx.col_np
        lam = ctx.lam
        v = ctx.v
        device = ctx.device
    
        # Step 1: dλ/dA = vv^T
        # A_grad = dlam * np.outer(v, v)

        A_grad_flat = 0.5 * dlam * (v[row_np] * v[col_np] + v[col_np] * v[row_np])

        if dv is not None and np.linalg.norm(dv) > 1e-12:
            # Constraint vector
            e_bk = np.zeros(n)
            e_bk[0] = 1.0

            # Top block (A - λI), shape: (n, n+2)
            I = sp.eye(n, format="csr")
            A_lamI = A - lam * I
            H_top = sp.hstack([A_lamI, -v.reshape(-1, 1), e_bk.reshape(-1, 1)])

            # Middle & bottom rows (1×(n+2))
            H_mid = sp.hstack([-v.reshape(1, -1), sp.csr_matrix((1, 2))])
            H_bot = sp.hstack([e_bk.reshape(1, -1), sp.csr_matrix((1, 2))])

            # Full KKT matrix: shape (n+2, n+2)
            H = sp.vstack([H_top, H_mid, H_bot]).tocsr()

            # RHS: shape (n+2, n)
            rhs = np.vstack([
                np.eye(n),
                np.zeros((2, n))
            ])

            # Solve H x = rhs
            X = spla.spsolve(H, rhs).reshape((n+2, n))
            u = X[:-2, :].sum(axis=0)

            # dΨ/dA = -½ (u ⊗ v + v ⊗ u)
            # dPsi_dA = -0.5 * (np.outer(u, v) + np.outer(v, u))
            dPsi_dA_flat = -0.5 * (u[row] * v[col] + v[row] * u[col])
            dangle = np.zeros(angle.shape[0])
            cosθ = np.cos(angle)
            sinθ = np.sin(angle)
            dA_dangle = 0.5 * np.hstack([
                -sinθ, -cosθ,
                cosθ, -sinθ,
                -sinθ, -cosθ,
                cosθ, -sinθ
            ]).reshape(8, -1)  # shape=(8, num_edges
            
            # Total gradient
            A_grad_flat += np.sum(dv * dPsi_dA_flat)
            dPsi_dA_grouped = dPsi_dA_flat.reshape(8, -1)  # shape=(8, num_edges)
            dangle = np.sum(dPsi_dA_grouped * dA_dangle, axis=0)  # shape=(num_edges,)
            dangle = torch.from_numpy(dangle).to(dtype).to(device)
            return dangle, None, None, None, None

# Alias
min_eig = MinEigenProblem.apply
min_eig_sparse = MinEigenProblemSparse.apply

if __name__ == "__main__":
    # ---------------------  TEST  --------------------
    # if A0  with n*n SPD matric
    import torch
    torch.manual_seed(0)
    # n = 4
    dtype = torch.float64
    # M  = torch.randn(n, n, dtype=dtype)
    # A0 = M @ M.T + 3.0 * torch.eye(n, dtype=dtype)        # SPD block
    npzDict = np.load('out.npz')
    A0 = npzDict['A']
    vW = npzDict['vW']
    iW = npzDict['iW']
    jW = npzDict['jW']
    n = A0.shape[0]
    A_sp = sp.coo_matrix((vW, (iW, jW)), shape=A0.shape)
    nonzero_per_row = np.bincount(A_sp.row) / 2
    diagA = np.concatenate((nonzero_per_row, np.array([0, 0])))
    A_sp = A_sp + sp.diags(diagA + 1e-5)
    A_coo = A_sp.tocoo()

    A = A_sp.todense()
    A = torch.from_numpy(A).to(dtype).requires_grad_(True)
    A_sparse = torch.sparse_coo_tensor((A_coo.row, A_coo.col), A_coo.data, A_coo.shape, dtype=dtype, requires_grad=True)
    '''
    eig_val, eig_vec = spla.eigsh(A_sp, k=1, return_eigenvectors=True, which='SA')
    print(eig_val)
    print(eig_vec)
    '''

    # ------- 2n×2n block‑diag(A0, A0) -------
    # A = torch.block_diag(A0, A0).requires_grad_(True)     # size 2n × 2n
    B = torch.eye(n, dtype=dtype)                         
    x0 = torch.randn(n, 1, dtype=dtype)                     # any vector

    from torch.autograd import grad       

    # 1) fp + bp, only for v
    lam, v = min_eig_sparse(A_sparse, x0)
    # lam.backward()
    vsum = v.sum()
    vsum.backward()
    grad_custom = A_sparse.grad.detach().clone()

    A.grad.zero_()
    defLam, defv = forward(A, B, x0)
    defVSum = defv.sum()
    defVSum.backward()
    grad_ref = A.grad.detach().clone()

    rel_err = (grad_custom - grad_ref).norm() / grad_ref.norm()
    print(f"grad_custom_max: {grad_custom.max()}")
    print(f"λ (one of the double roots) = {lam.item():.6f}")
    print(f"‖grad_custom - grad_ref‖ / ‖grad_ref‖ = {rel_err:.3e}")
