import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from module.gridMesher import GridMesh


class H8_K_XFE:
    def __init__(self, device=torch.device('cuda'), **kwargs):
        # Material properties
        self.E = 1.0 if 'E' not in kwargs else kwargs['E']  # Young's modulus
        self.nu = 0.3 if 'nu' not in kwargs else kwargs['nu']  # Poisson's ratio
        
        # Compute shear modulus
        self.G = self.E / (2 * (1 + self.nu))
        
        # Compute compliance matrix
        C = np.array([
            [1/self.E, -self.nu/self.E, -self.nu/self.E, 0, 0, 0],
            [-self.nu/self.E, 1/self.E, -self.nu/self.E, 0, 0, 0],
            [-self.nu/self.E, -self.nu/self.E, 1/self.E, 0, 0, 0],
            [0, 0, 0, 1/self.G, 0, 0],
            [0, 0, 0, 0, 1/self.G, 0],
            [0, 0, 0, 0, 0, 1/self.G]
        ])
        self.C_inv = torch.tensor(np.linalg.inv(C), dtype=torch.float32, device=device)
        
        # Gauss integration points and weights
        _sqrt_3_5 = math.sqrt(3/5)
        integration_point = torch.tensor([-_sqrt_3_5, 0, _sqrt_3_5], dtype=torch.float32) / 2
        integration_weight = torch.tensor([5/9, 8/9, 5/9], dtype=torch.float32) / 2
        
        # Generate all integration points
        all_integration_points = np.vstack(
            np.meshgrid(integration_point, integration_point, integration_point)
        ).reshape(3, -1).T
        
        # Compute integration weights for all points
        int_weight = np.prod(
            np.meshgrid(integration_weight, integration_weight, integration_weight)
        ).flatten()
        
        self.int_weight = torch.tensor(int_weight, device=device)
        self.B = torch.tensor(self.matrixB(all_integration_points), dtype=torch.float32, device=device)

    def matrixB(self, xyz):
        """Compute strain-displacement matrix B for H8 element"""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        o = np.zeros_like(x)
        
        # Shape function derivatives
        b = (-(0.5 - y) * (0.5 - z), o, o, (0.5 - y) * (0.5 - z), o, o,
             (0.5 - z) * (y + 0.5), o, o, -(0.5 - z) * (y + 0.5), o, o,
             -(0.5 - y) * (z + 0.5), o, o, (0.5 - y) * (z + 0.5), o, o,
             (y + 0.5) * (z + 0.5), o, o, -(y + 0.5) * (z + 0.5), o, o,
             o, -(0.5 - x) * (0.5 - z), o, o, -(0.5 - z) * (x + 0.5), o,
             o, (0.5 - z) * (x + 0.5), o, o, (0.5 - x) * (0.5 - z), o,
             o, -(0.5 - x) * (z + 0.5), o, o, -(x + 0.5) * (z + 0.5), o,
             o, (x + 0.5) * (z + 0.5), o, o, (0.5 - x) * (z + 0.5), o,
             o, o, -(0.5 - x) * (0.5 - y), o, o, -(0.5 - y) * (x + 0.5),
             o, o, -(x + 0.5) * (y + 0.5), o, o, -(0.5 - x) * (y + 0.5),
             o, o, (0.5 - x) * (0.5 - y), o, o, (0.5 - y) * (x + 0.5),
             o, o, (x + 0.5) * (y + 0.5), o, o, (0.5 - x) * (y + 0.5),
             o, -(0.5 - x) * (0.5 - y), -(0.5 - x) * (0.5 - z), o, -(0.5 - y) * (x + 0.5), -(0.5 - z) * (x + 0.5),
             o, -(x + 0.5) * (y + 0.5), (0.5 - z) * (x + 0.5), o, -(0.5 - x) * (y + 0.5), (0.5 - x) * (0.5 - z),
             o, (0.5 - x) * (0.5 - y), -(0.5 - x) * (z + 0.5), o, (0.5 - y) * (x + 0.5), -(x + 0.5) * (z + 0.5),
             o, (x + 0.5) * (y + 0.5), (x + 0.5) * (z + 0.5), o, (0.5 - x) * (y + 0.5), (0.5 - x) * (z + 0.5),
             -(0.5 - x) * (0.5 - y), o, -(0.5 - y) * (0.5 - z), -(0.5 - y) * (x + 0.5), o, (0.5 - y) * (0.5 - z),
             -(x + 0.5) * (y + 0.5), o, (0.5 - z) * (y + 0.5), -(0.5 - x) * (y + 0.5), o, -(0.5 - z) * (y + 0.5),
             (0.5 - x) * (0.5 - y), o, -(0.5 - y) * (z + 0.5), (0.5 - y) * (x + 0.5), o, (0.5 - y) * (z + 0.5),
             (x + 0.5) * (y + 0.5), o, (y + 0.5) * (z + 0.5), (0.5 - x) * (y + 0.5), o, -(y + 0.5) * (z + 0.5),
             -(0.5 - x) * (0.5 - z), -(0.5 - y) * (0.5 - z), o, -(0.5 - z) * (x + 0.5), (0.5 - y) * (0.5 - z), o,
             (0.5 - z) * (x + 0.5), (0.5 - z) * (y + 0.5), o, (0.5 - x) * (0.5 - z), -(0.5 - z) * (y + 0.5), o,
             -(0.5 - x) * (z + 0.5), -(0.5 - y) * (z + 0.5), o, -(x + 0.5) * (z + 0.5), (0.5 - y) * (z + 0.5), o,
             (x + 0.5) * (z + 0.5), (y + 0.5) * (z + 0.5), o, (0.5 - x) * (z + 0.5), -(y + 0.5) * (z + 0.5), o)
        
        return np.stack(b, -1).reshape((xyz.shape[0], 6, 24))

    def compute_Ke(self, density, density_penal=3):
        """Compute element stiffness matrix for H8 element with XFEM"""
        weight, B = self.int_weight, self.B
        
        # Compute B^T * C * B for each integration point
        BT_C_B = torch.einsum('d,dji,jk,dkl->dil', weight, B, self.C_inv, B)
        
        # Apply density penalization
     