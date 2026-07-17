import numpy as np
import torch

from module.minEigenProblem import min_eig

class StripeGeneration:
    def __init__(self, gridSize:list|tuple):
        self.gridSize = gridSize
        points = np.array(gridSize)
        self.pre_compute()

    def pre_compute(self):
        self.points, self.edges = self._edges()
        I, J = self.edges[:, 0]*2, self.edges[:, 1]*2
        self.col = np.hstack([I, I, I + 1, I + 1, J, J, J + 1, J + 1]).flatten()
        self.row = np.hstack([J, J + 1, J, J + 1, I, I + 1, I, I + 1]).flatten()

    def _edges(self):
        nelx, nely, nelz = self.gridSize
        nx, ny, nz = nelx + 1, nely + 1, nelz + 1

        # Create a 3D array of node IDs
        node_ids = np.arange(nx * ny * nz).reshape((nx, ny, nz))
        node_pos = np.array([[i, j, k] for i in range(nx) for j in range(ny) for k in range(nz)])
        points = node_pos
        edges = []

        # X-direction edges
        if nx > 1:
            n1 = node_ids[:-1, :, :].ravel()
            n2 = node_ids[1:, :, :].ravel()
            edges.append(np.stack([n1, n2], axis=1))

        # Y-direction edges
        if ny > 1:
            n1 = node_ids[:, :-1, :].ravel()
            n2 = node_ids[:, 1:, :].ravel()
            edges.append(np.stack([n1, n2], axis=1))

        # Z-direction edges
        if nz > 1:
            n1 = node_ids[:, :, :-1].ravel()
            n2 = node_ids[:, :, 1:].ravel()
            edges.append(np.stack([n1, n2], axis=1))

        # Concatenate all edges
        if edges:
            edges = np.concatenate(edges, axis=0)
        else:
            edges = np.empty((0, 2), dtype=int)

        return points, edges

    def vectorField2Stripe(self, vectorField:np.array):
        # vectorField: (N, 3)
        assert vectorField.shape[1] == 3, "vectorField must be a 3D array"
        nv = vectorField.shape[0]

        edges = self.edges
        # edges: (N, 2) [[e1, e2];[e3, e4];...]
        # calculate the angle via the edge
        Vs = vectorField[edges[:, 0]]
        Ve = vectorField[edges[:, 1]]
        angle = np.arccos(np.sum(Vs * Ve, axis=1) / (np.linalg.norm(Vs, axis=1) * np.linalg.norm(Ve, axis=1)))
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        # calculate the stripe
        # generate the A matrix
        row, col = self.row, self.col
        val = 0.5 * np.hstack([cos_angle,-sin_angle,sin_angle,cos_angle,cos_angle,-sin_angle,sin_angle,cos_angle]).flatten()
        A = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), val, size=(vectorField.shape[0]*2, vectorField.shape[0]*2))
        B = torch.sparse_
        lam, v = min_eig(A, B, x0)
        x0 = torch.randn(2*n, 1, dtype=dtype)                 # 任意初始向量

def test_vectorField(pos):
    newpos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-6)
    return newpos

if __name__ == "__main__":
    stripeGeneration = StripeGeneration((10, 10, 10))
    points = stripeGeneration.points
    print(stripeGeneration.edges)
    vectorField = test_vectorField(points)
    stripeGeneration.vectorField2Stripe(vectorField)