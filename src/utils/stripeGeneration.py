import numpy as np
import torch
from math import sin, cos, acos, tan, pi, floor, ceil
import openmesh
from abc import ABC
import pyvista as pv
from scipy.spatial import cKDTree
import os
import scipy.spatial.distance
import scipy
import potpourri3d as pp3d

import mkl
import pyMKL


###########
def savePath2Obj(filePath, mesh: pv.PolyData):
    with open(filePath, 'w') as f:
        for p in mesh.points:
            f.write('v {} {} {}\n'.format(p[0], p[1], p[2]))

        L = mesh.lines.reshape(-1, 3)[:, 1:]
        for l in L:
            f.write('l {} {}\n'.format(l[0] + 1, l[1] + 1))


def savePath2ObjExt(filePath, mesh: pv.PolyData, lineinFaceIdx):
    with open(filePath, 'w') as f:
        for p in mesh.points:
            f.write('v {} {} {}\n'.format(p[0], p[1], p[2]))

        L = mesh.lines.reshape(-1, 3)[:, 1:]
        for l in L:
            f.write('l {} {}\n'.format(l[0] + 1, l[1] + 1))

        for lif in lineinFaceIdx:
            f.write('#li {}\n'.format(lif + 1))


###########
def IPM(A, B, x0):
    return inversePowerMethod.apply(A, B, x0)


class inversePowerMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, x0):
        try:
            iA, jA = A._indices().detach().numpy()
            vA = A._values()
            sA = scipy.sparse.coo_matrix((vA, (iA, jA)))
            factor = pyMKL.pardisoSolver((sA).astype(np.float64), 2)

            B = B.diag().detach().numpy()
            x0 = x0.detach().numpy()
            factor.run_pardiso(12)
            B_cpu = factor.run_pardiso(33, (B*x0).astype(np.float64))
            if np.isnan(B_cpu[0]):
                factor.run_pardiso(-1)
                raise Exception("Nan happens when solving Linear system!")
        except:
            print('L is not SPD!')
            if torch.isnan(A).any():
                print('   Checked: A contains Nan!')
                raise ('L include Nan!')

        PsiList = []
        Psi = x0.copy()
        PsiList.append(Psi.copy())
        for _ in range(20):
            Psi = factor.run_pardiso(33, (B * Psi).astype(np.float64))
            PsiList.append(Psi.copy())
            Psi = Psi / (np.sqrt(Psi.T @ (B * Psi)))
            PsiList.append(Psi.copy())

        ctx.A = sA
        ctx.B = B
        ctx.factor = factor
        ctx.psiList = PsiList
        return Psi

    @staticmethod
    def backward(ctx, grad_psi):
        A = ctx.A
        B = ctx.B
        Psi = ctx.psiList
        factor = ctx.factor

        A_grad = torch.zeros_like(A)
        iteration_times = (len(Psi) - 1) // 2
        grad_psi_2n = grad_psi
        for i in range(iteration_times):
            n = 2 * i
            xbx = Psi[-n - 2] * B * Psi[-n - 2]
            # Psi_{2n} => Psi_{2n-1} => Psy_{2n} = Psy_{2n-1} / sqrt(Psy_{2n-1}.T @ B @ Psy_{2n-1})
            grad_Psi_2nm1 = xbx ** (-0.5) * grad_psi_2n - xbx ** (-1.5) * Psi[-n - 2] * \
                            torch.dot((B * Psi[-n - 2]).flatten(), grad_psi_2n.flatten())

            # Psi_{2n-1} => Psi_{2n} => Psy_{2n-1} = A^-1 @ B @ Psy_{2n-2}
            grad_Psi2nm1_2nm2 = factor.run_pardiso(33, (B * grad_Psi_2nm1).astype(np.float64))

            # Psi_{2n-1} => A => Psy_{2n-1} = Psy_{2n-1} = A^-1 @ B @ Psy_{2n-2}
            grad_Psi2nm1_A_half = torch.outer(factor.run_pardiso(33, (B * grad_Psi_2nm1).astype(np.float64)).flatten(), Psi[-n - 2].flatten())
            grad_Psi2nm1_A = -(grad_Psi2nm1_A_half + grad_Psi2nm1_A_half.T) / 2

            grad_psi_2n = grad_Psi2nm1_2nm2
            A_grad += grad_Psi2nm1_A
        return A_grad, None, None


###########


def arg(c):
    return np.arctan2(c.imag, c.real)


def fmodPI(theta):
    return theta - (2. * pi) * floor((theta + pi) / (2. * pi))


def delete_isolated_vertices(mesh_om: openmesh.TriMesh):
    for vit in mesh_om.vertices():
        if not mesh_om.halfedge_handle(vit).is_valid():
            mesh_om.delete_vertex(vit)
    mesh_om.garbage_collection()


def extend_mesh_boundary(mesh, extend_length=0.1):
    f = mesh.faces.reshape((-1, 4))[:, 1:]
    mesh_om = openmesh.TriMesh(mesh.points, f)
    n_halfedge = mesh_om.n_halfedges()
    heit_visited = np.zeros(n_halfedge)

    # extend_length = 0.01
    for i in range(n_halfedge):
        heit = mesh_om.halfedge_handle(i)
        if heit_visited[heit.idx()]:
            continue

        heit_visited[heit.idx()] = True
        if mesh_om.is_boundary(heit):
            new_v_line = []
            vstart = mesh_om.from_vertex_handle(heit)
            while True:
                oppo_he = mesh_om.opposite_halfedge_handle(heit)
                face_normal = mesh_om.calc_face_normal(mesh_om.face_handle(oppo_he))

                vs = mesh_om.from_vertex_handle(heit)
                ve = mesh_om.to_vertex_handle(heit)
                length = mesh_om.calc_edge_length(heit)
                mid_point = (mesh_om.point(vs) + mesh_om.point(ve)) / 2
                direction = mesh_om.point(ve) - mesh_om.point(vs)
                extend_vec = np.cross(face_normal, direction)
                extend_vec /= np.linalg.norm(extend_vec, keepdims=True)
                v_new = mid_point + extend_vec * length * extend_length - face_normal * length * 1e-4

                new_vectex_handle = mesh_om.add_vertex(v_new)

                mesh_om.add_face([vs, ve, new_vectex_handle])

                new_v_line.append(vs)
                new_v_line.append(new_vectex_handle)

                # next -> oppo -> next
                heit = mesh_om.next_halfedge_handle(heit)
                heit = mesh_om.opposite_halfedge_handle(heit)
                v_next = mesh_om.to_vertex_handle(heit)

                heit = mesh_om.next_halfedge_handle(heit)

                # print(vs.idx())
                if v_next == vstart:
                    new_v_line.append(new_v_line[0])
                    new_v_line.append(new_v_line[1])
                    break
                heit_visited[heit.idx()] = True

            for i in range(1, len(new_v_line) // 2):
                mesh_om.add_face([new_v_line[2 * i - 1], new_v_line[2 * i], new_v_line[2 * i + 1]])

    mesh_om.garbage_collection()
    delete_isolated_vertices(mesh_om)

    mesh_new = pv.make_tri_mesh(mesh_om.points(), mesh_om.fv_indices())
    return mesh_new


def getClosestPoints(mesh, tolerance=1e-4):
    kdtree = cKDTree(mesh.points)
    pairs = kdtree.query_pairs(r=tolerance, output_type='ndarray')
    return pairs


def pointInFan(P, A, B, C):
    # first project P into tri ABC, get Barycentric coordinate (u,v)
    # then check u, v and the angle of A-projP A-B

    # AP = u * AB + v* AC

    v0 = B - A
    v1 = C - A
    v2 = P - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    v = (dot11 * dot02 - dot01 * dot12) * invDenom
    w = (dot00 * dot12 - dot01 * dot02) * invDenom
    u = 1.0 - v - w

    projP = u * A + v * B + w * C
    A_projP = projP - A
    A_projP_norm = np.linalg.norm(A_projP)
    v0_norm = np.linalg.norm(v0)
    cos_theta = max(min(np.dot(A_projP, v0) / (A_projP_norm * v0_norm + 1e-9), 1), 0)
    tf = (w >= -1e-8) and (v >= -1e-8) and (u <= 1)
    # print(u, v, acos(np.dot(projP-A, v1)/A_projP_norm), A_projP_norm * v2_norm)
    return tf, acos(cos_theta)


class DiffStripeGenerationDoubleCover():
    def __init__(self, mesh: pv.UnstructuredGrid, frequence=3):
        self.pvmesh = mesh
        f = mesh.faces.reshape((-1, 4))[:, 1:]
        self.mesh = openmesh.TriMesh(mesh.points, f)
        self.v = frequence
        # self._generateTheta()
        self.points = torch.from_numpy(mesh.points).float()
        self.faces = torch.from_numpy(f.copy()).long()

        all_faces = np.zeros(self.faces.shape[0], dtype=int)
        for fit in self.mesh.faces():
            if self.mesh.is_boundary(fit):
                all_faces[fit.idx()] = 1

        self.is_boundary_faces = torch.tensor(all_faces).flatten().long()
        self.non_boundary_faces_idx = torch.tensor(np.argwhere(all_faces == 1)).flatten().long()
        self.boundary_faces_idx = torch.tensor(np.argwhere(all_faces == 0)).flatten().long()

        # if extend faces, boundary faces is the 1-ring of real boundary
        self.is_boundary_vertices = torch.zeros(self.points.shape[0]).long()
        self.is_boundary_edges = torch.zeros(self.mesh.n_edges()).long()

        for vit in self.mesh.vertices():
            if self.mesh.is_boundary(vit):
                self.is_boundary_vertices[vit.idx()] = 1
                for vvit in self.mesh.vv(vit):
                    self.is_boundary_vertices[vvit.idx()] = 1

        # remove virtual boundary
        for vit in self.mesh.vertices():
            if self.mesh.is_boundary(vit):
                self.is_boundary_vertices[vit.idx()] = 0

        for vit in self.mesh.vertices():
            if self.mesh.is_boundary(vit):
                for vheit in self.mesh.voh(vit):
                    if self.mesh.is_boundary(vheit):
                        continue
                    opposite_heit = self.mesh.next_halfedge_handle(vheit)
                    if self.is_boundary_vertices[self.mesh.from_vertex_handle(opposite_heit).idx()] == 1 and \
                            self.is_boundary_vertices[self.mesh.to_vertex_handle(opposite_heit).idx()] == 1:
                        self.is_boundary_edges[self.mesh.edge_handle(opposite_heit).idx()] = 1

        self.boundary_vertices_idx = torch.argwhere(self.is_boundary_vertices == 1).flatten().long()
        self.non_boundary_vertices_idx = torch.argwhere(self.is_boundary_vertices == 0).flatten().long()
        self.boundary_edges_idx = torch.argwhere(self.is_boundary_edges == 1).flatten().long()

        self.paramIndex = torch.zeros(self.mesh.n_faces())
        self.partBoundary_vertices_idx = None
        self.doublecover_point_idx = None
        self.sij = None
        self.ignoredFlipFaces = None
        # main process
        ## first part: solve \Phi
        # get \psi from optimizator
        # get \omega_ij
        # get L = {sin omega ; cos omega}
        # A = L[W] dynamic index, W is const matrix
        # B is const
        # solve Psi as x in min x^TAx x^TBx=1

    def preprocess(self, initTheta=True):
        num_vertiex = self.mesh.n_vertices()
        if initTheta:
            self.theta = torch.from_numpy(self.generateTheta()).float()
            self.thetaIJ = torch.from_numpy(self.mesh.halfedge_property_array('theta')).float()
        self.area = torch.from_numpy(self.generateArea()).float()
        self.edgeLength = torch.from_numpy(self.generateEdgeLength()).float()

        weight, diagA = self.generateCotWeight()
        self.weight = torch.from_numpy(weight).float()
        self.diagA = torch.from_numpy(diagA).float()

        self.B = torch.from_numpy(self.generateMatrixB()).float()
        self.defaultX = torch.randn(num_vertiex * 2, 1, dtype=torch.float32)
        # self.W_th = torch.from_numpy(self.generateMatrixW()).long()
        self.generateSparseW()

    def solvePsi(self, omega: torch.Tensor):
        assert omega.shape[0] == self.mesh.n_edges()
        A = self.generateSparseA(omega)
        B = torch.diag(self.B)
        # _, Psi = calculateMinEigen(A, B, self.defaultX)
        Psi = IPM(A, B, self.defaultX)
        return Psi

    # need more think
    def solvePsi_withBoundary(self, omega):
        assert omega.shape[0] == self.mesh.n_edges()
        A = self.generateSparseA(omega)
        # B = torch.diag(self.B)
        # min x^tAx
        # s.t x_I = 1, x_{I+1} = 0

        bv_size = self.boundary_vertices_idx.shape[0]
        I = torch.zeros((bv_size, self.points.shape[0] * 2))
        I_range = torch.arange(0, bv_size, dtype=torch.long)

        Ip1 = I.detach().clone()
        I[I_range, 2 * self.boundary_vertices_idx] = 1
        Ip1[I_range, 2 * self.boundary_vertices_idx + 1] = 1

        bv_size_zero = torch.zeros((bv_size, bv_size))
        bv_size_zero_row = torch.zeros(bv_size)
        A_size_zero_row = torch.zeros(self.points.shape[0] * 2)

        H0 = torch.cat((A, I, Ip1), dim=0)
        H1 = torch.cat((I.T, bv_size_zero, bv_size_zero), dim=0)
        H2 = torch.cat((Ip1.T, bv_size_zero, bv_size_zero), dim=0)
        H = torch.cat((H0, H1, H2), dim=1)

        b = torch.cat((A_size_zero_row, bv_size_zero_row, bv_size_zero_row + 1))
        Psi = torch.linalg.solve(H, b)

        return Psi[:self.points.shape[0] * 2]

    def solvePsi_withPart_Boundary_old(self, omega):
        assert omega.shape[0] == self.mesh.n_edges()
        # A = self.generateMatrixA(omega)
        A = self.generateSparseA(omega).to_dense()

        partBoundary_vertices_idx = torch.tensor([0]).long()

        bv_size = partBoundary_vertices_idx.shape[0]
        I = torch.zeros((bv_size, self.points.shape[0] * 2))
        I_range = torch.arange(0, bv_size, dtype=torch.long)

        Ip1 = I.detach().clone()
        I[I_range, 2 * partBoundary_vertices_idx] = 1
        Ip1[I_range, 2 * partBoundary_vertices_idx + 1] = 1

        bv_size_zero_row = torch.zeros(bv_size)
        A_size_zero_row = torch.zeros(self.points.shape[0] * 2)

        H0 = torch.cat((A, I, Ip1), dim=0)
        H1 = torch.cat((I.T, torch.zeros((bv_size * 2, bv_size))), dim=0)
        H2 = torch.cat((Ip1.T, torch.zeros((bv_size * 2, bv_size))), dim=0)

        H = torch.cat((H0, H1, H2), dim=1)
        b = torch.cat((A_size_zero_row, bv_size_zero_row, bv_size_zero_row + 1))
        b[partBoundary_vertices_idx] = 1

        Psi = torch.linalg.solve(H, b)
        return Psi[:(self.points.shape[0] * 2)]

    def solvePsi_withPart_Boundary_old(self, omega):
        assert omega.shape[0] == self.mesh.n_edges()

        cosOmega = torch.cos(omega) * self.weight
        sinOmega = torch.sin(omega) * self.weight
        vW = torch.hstack((-cosOmega, -sinOmega * self.sij, sinOmega, -cosOmega * self.sij,
                           -cosOmega, sinOmega * self.sij, -sinOmega, -cosOmega * self.sij)).flatten()
        # vW = torch.hstack((-cosOmega, -sinOmega, sinOmega, -cosOmega,
        #                    -cosOmega, sinOmega, -sinOmega, -cosOmega)).flatten()
        nRange = torch.arange(2 * self.mesh.n_vertices())
        partBoundary_vertices_idx = torch.tensor([0]).long()

        '''
        A[partBoundary_vertices_idx*2, 2 * self.mesh.n_vertices()] = 1
        A[partBoundary_vertices_idx*2+1, 2 * self.mesh.n_vertices()+1] = 1
        A[2 * self.mesh.n_vertices(), partBoundary_vertices_idx*2] = 1
        A[2 * self.mesh.n_vertices()+1, 1] = 1
        '''

        iH = torch.tensor([partBoundary_vertices_idx * 2, partBoundary_vertices_idx * 2 + 1, 2 * self.mesh.n_vertices(), 2 * self.mesh.n_vertices() + 1])
        jH = torch.tensor([2 * self.mesh.n_vertices(), 2 * self.mesh.n_vertices()+1, partBoundary_vertices_idx * 2, partBoundary_vertices_idx*2 + 1])
        vH = torch.tensor([1, 1, 1, 1])*0.1
        '''
        H = torch.sparse_coo_tensor(torch.vstack((iH, jH)), vH)
        A = H + \
            torch.sparse_coo_tensor(torch.vstack((self.iW, self.jW)), vW, size=(2 * self.mesh.n_vertices()+2, 2 * self.mesh.n_vertices()+2)) + \
            torch.sparse_coo_tensor(torch.vstack((nRange, nRange)), self.diagA + 1e-4, size=(2 * self.mesh.n_vertices()+2, 2 * self.mesh.n_vertices()+2))
        '''
        I = torch.cat((self.iW, nRange, iH), dim=0).detach().numpy()
        J = torch.cat((self.jW, nRange, jH), dim=0).detach().numpy()
        V = torch.cat((vW, self.diagA + 1e-4, vH), dim=0).detach().numpy()

        bv_size = partBoundary_vertices_idx.shape[0]
        bv_size_zero_row = torch.zeros(bv_size)
        A_size_zero_row = torch.zeros(self.points.shape[0] * 2)
        b = torch.cat((A_size_zero_row, bv_size_zero_row, bv_size_zero_row + 1))
        b[2*partBoundary_vertices_idx+1] = 1

        A = scipy.sparse.csc_matrix((V, (I, J)))
        Psi = scipy.sparse.linalg.spsolve(A, b)
        Psi = torch.tensor(Psi)

        if Psi.reshape(-1,2).abs().sum(1).min() <= 0:
            newA = scipy.sparse.csc_matrix((vW, (self.iW.detach().numpy(), self.jW.detach().numpy()))) + \
                   scipy.sparse.csc_matrix((self.diagA.detach().numpy() + 1e-4, (nRange.detach().numpy(), nRange.detach().numpy())))
            Psi = np.real(scipy.sparse.linalg.eigs(newA, 1, which='SM')[1])
            return Psi

        return Psi[:(self.points.shape[0] * 2)]

    def solvePsi_withPart_Boundary(self, omega):
        assert omega.shape[0] == self.mesh.n_edges()

        cosOmega = torch.cos(omega) * self.weight
        sinOmega = torch.sin(omega) * self.weight
        vW = torch.hstack((-cosOmega, -sinOmega * self.sij, sinOmega, -cosOmega * self.sij,
                           -cosOmega, sinOmega * self.sij, -sinOmega, -cosOmega * self.sij)).flatten()

        nRange = torch.arange(2 * self.mesh.n_vertices())

        newA = scipy.sparse.csc_matrix((vW, (self.iW.detach().numpy(), self.jW.detach().numpy()))) + \
               scipy.sparse.csc_matrix((self.diagA.detach().numpy() + 1e-4, (nRange.detach().numpy(), nRange.detach().numpy())))
        # Psi = np.real(scipy.sparse.linalg.eigs(newA, 1, which='SM')[1])
        Psi = np.real(scipy.sparse.linalg.eigs(newA, 1, sigma=0, which='LM')[1])
        return Psi

    def solvePsi_LSQ(self, omega, useBoundary=False):
        assert omega.shape[0] == self.mesh.n_edges()
        sizeA = 2 * self.mesh.n_vertices()
        cosOmega = torch.cos(omega) * self.weight
        sinOmega = torch.sin(omega) * self.weight
        vW = torch.hstack((-cosOmega, -sinOmega * self.sij, sinOmega, -cosOmega * self.sij,
                           -cosOmega, sinOmega * self.sij, -sinOmega, -cosOmega * self.sij)).flatten()
        # vW = torch.hstack((-cosOmega, -sinOmega, sinOmega, -cosOmega,
        #                    -cosOmega, sinOmega, -sinOmega, -cosOmega)).flatten()
        nRange = torch.arange(2 * self.mesh.n_vertices())
        '''
        A = torch.sparse_coo_tensor(torch.vstack((self.iW, self.jW)), vW) + \
            torch.sparse_coo_tensor(torch.vstack((nRange, nRange)), self.diagA + 1e-4)
        '''
        if useBoundary:
            fixed_vectices = self.boundary_vertices_idx
        else:
            fixed_vectices = torch.tensor([0]).long()
        bv_size = fixed_vectices.shape[0]

        I_range = torch.arange(0, bv_size, dtype=torch.long)

        # I => [I_range, 2 * fixed_vectices, 1]
        # Ip1 => [I_range, 2 * fixed_vectices + 1]
        iI = I_range + sizeA
        iIp1 = I_range + sizeA + bv_size
        jI = 2 * fixed_vectices
        jIp1 = 2 * fixed_vectices + 1

        iH = torch.cat((self.iW, iI, iIp1, jI, jIp1), dim=0)
        jH = torch.cat((self.jW, jI, jIp1, iI, iIp1), dim=0)
        vH = torch.cat((self.vW, torch.ones(4 * bv_size)), dim=0)

        H = torch.sparse_coo_tensor(torch.vstack((iH, jH)), vH) + \
            torch.sparse_coo_tensor(torch.vstack((nRange, nRange)), self.diagA + 1e-4)

        bv_size_zero_row = torch.zeros(bv_size)
        A_size_zero_row = torch.zeros(sizeA * 2)
        b = torch.cat((A_size_zero_row, bv_size_zero_row, bv_size_zero_row + 1))
        Psi = torch.sparse.linalg.solve(H, b)

        return Psi[:(self.points.shape[0] * 2)]

    def generateEdgeLength(self):
        num_edges = self.mesh.n_edges()
        edgeLength = np.zeros(num_edges)
        for eit in self.mesh.edges():
            edgeLength[eit.idx()] = self.mesh.calc_edge_sqr_length(eit)
        self.mesh.set_edge_property_array('length', edgeLength)
        return edgeLength

    def generateArea(self):
        num_faces = self.mesh.n_faces()
        area = np.zeros(num_faces)

        for fit in self.mesh.faces():
            area[fit.idx()] = self.mesh.calc_sector_area(
                self.mesh.halfedge_handle(fit))
        self.mesh.set_face_property_array('area', area)
        return area

    def generateCotWeight(self):
        number_vertex = self.mesh.n_vertices()
        num_edges = self.mesh.n_edges()
        w = np.zeros(num_edges)
        diagA = np.zeros((2 * number_vertex))

        for eit in self.mesh.edges():
            he0 = self.mesh.halfedge_handle(eit, 0)
            he1 = self.mesh.halfedge_handle(eit, 1)

            vs = self.mesh.from_vertex_handle(he0)
            ve = self.mesh.to_vertex_handle(he0)

            he0_next = self.mesh.next_halfedge_handle(he0)
            he1_next = self.mesh.next_halfedge_handle(he1)

            cot0 = max(1 / tan(self.mesh.calc_sector_angle(he0_next)), 1e-7)
            cot1 = max(1 / tan(self.mesh.calc_sector_angle(he1_next)), 1e-7)

            if self.mesh.is_boundary(he0):
                cot0 = 0

            if self.mesh.is_boundary(he1):
                cot1 = 0

            w[eit.idx()] = 0.5 * (cot0 + cot1)

            i = 2 * vs.idx()
            j = 2 * ve.idx()

            diagA[i] += w[eit.idx()]
            diagA[i + 1] += w[eit.idx()]
            diagA[j] += w[eit.idx()]
            diagA[j + 1] += w[eit.idx()]
        return w, diagA

    def generateMatrixA(self, omega: torch.Tensor):
        assert omega.shape[0] == self.mesh.n_edges()
        cosOmega = torch.cos(omega) * self.weight
        sinOmega = torch.sin(omega) * self.weight
        L = torch.hstack((cosOmega, -cosOmega, sinOmega, -sinOmega, torch.tensor(0)))
        A = L[self.W_th] + torch.eye(self.W_th.shape[0]) * 1e-4 + torch.diag(self.diagA)
        return A

    def generateSparseA(self, omega: torch.Tensor):
        assert omega.shape[0] == self.mesh.n_edges()
        cosOmega = torch.cos(omega) * self.weight
        sinOmega = torch.sin(omega) * self.weight
        vW = torch.hstack((-cosOmega, -sinOmega * self.sij, sinOmega, -cosOmega * self.sij,
                           -cosOmega, sinOmega * self.sij, -sinOmega, -cosOmega * self.sij)).flatten()
        # vW = torch.hstack((-cosOmega, -sinOmega, sinOmega, -cosOmega,
        #                    -cosOmega, sinOmega, -sinOmega, -cosOmega)).flatten()
        nRange = torch.arange(2 * self.mesh.n_vertices())
        A = torch.sparse_coo_tensor(torch.vstack((self.iW, self.jW)), vW) + \
            torch.sparse_coo_tensor(torch.vstack((nRange, nRange)), self.diagA + 1e-4)
        return A

    def generateSparseW(self):
        num_vertices = self.mesh.n_vertices()
        num_edges = self.mesh.n_edges()
        ev = self.mesh.ev_indices()
        vs, ve = ev[:, 0], ev[:, 1]
        I, J = 2 * vs, 2 * ve

        neRange = np.arange(0, num_edges)

        iW = np.hstack([I, I, I + 1, I + 1, J, J, J + 1, J + 1]).flatten()
        jW = np.hstack([J, J + 1, J, J + 1, I, I + 1, I, I + 1]).flatten()

        # vW = np.array([num_edges * 1, num_edges * 3, num_edges * 2, num_edges * 1,
        #                num_edges * 1, num_edges * 2, num_edges * 3, num_edges * 1]) + neRange
        self.iW = torch.tensor(iW, dtype=torch.long)
        self.jW = torch.tensor(jW, dtype=torch.long)
        return (self.iW, self.jW)

    def generateMatrixW(self):
        num_vertices = self.mesh.n_vertices()
        num_edges = self.mesh.n_edges()

        W = np.zeros((num_vertices * 2, num_vertices * 2), dtype=int) - 1

        for eit in self.mesh.edges():
            he0 = self.mesh.halfedge_handle(eit, 0)
            vs = self.mesh.from_vertex_handle(he0)
            ve = self.mesh.to_vertex_handle(he0)

            i = 2 * vs.idx()
            j = 2 * ve.idx()

            W[i + 0, j + 0] = num_edges * 1 + eit.idx()
            W[i + 0, j + 1] = num_edges * 3 + eit.idx()
            W[i + 1, j + 0] = num_edges * 2 + eit.idx()
            W[i + 1, j + 1] = num_edges * 1 + eit.idx()

            W[j + 0, i + 0] = num_edges * 1 + eit.idx()
            W[j + 0, i + 1] = num_edges * 2 + eit.idx()
            W[j + 1, i + 0] = num_edges * 3 + eit.idx()
            W[j + 1, i + 1] = num_edges * 1 + eit.idx()

        return W

    def generateMatrixB(self):
        num_vertices = self.mesh.n_vertices()
        B = np.zeros(num_vertices * 2)
        area = self.mesh.face_property_array('area')
        for fit in self.mesh.faces():
            v = self.mesh.fv_indices()[fit.idx()]
            _area = area[fit.idx()]
            B[2 * v[0]] += 1 / 3 * _area
            B[2 * v[0] + 1] += 1 / 3 * _area
            B[2 * v[1]] += 1 / 3 * _area
            B[2 * v[1] + 1] += 1 / 3 * _area
            B[2 * v[2]] += 1 / 3 * _area
            B[2 * v[2] + 1] += 1 / 3 * _area
        return B

    def generateTheta(self):
        num_vertices = self.mesh.n_vertices()

        v_start_he = np.zeros(self.mesh.n_vertices(), dtype=int)
        # set start halfedge of each vertex at v_start_he
        for vit in self.mesh.vertices():
            # check vit if is boundary
            if self.mesh.is_boundary(vit):
                he = self.mesh.halfedge_handle(vit)
                oppo_he = self.mesh.opposite_halfedge_handle(he)
                # ccw iterator halfedge
                while he.is_valid() and (self.mesh.face_handle(oppo_he).idx() != -1):
                    he = self.mesh.next_halfedge_handle(self.mesh.opposite_halfedge_handle(he))
                    oppo_he = self.mesh.opposite_halfedge_handle(he)

                v_start_he[vit.idx()] = he.idx()
            else:
                he = self.mesh.halfedge_handle(vit)
                v_start_he[vit.idx()] = he.idx()
        self.mesh.set_vertex_property_array('x_axis', v_start_he)

        theta = np.zeros(self.mesh.n_halfedges())
        for vit in self.mesh.vertices():
            he_idx = v_start_he[vit.idx()]
            he = self.mesh.halfedge_handle(he_idx)
            vs = self.mesh.point(vit)
            while True:
                ve = self.mesh.point(self.mesh.to_vertex_handle(he))
                theta[he.idx()], _ = self._tangent2angle(ve - vs, vit.idx())
                he = self.mesh.opposite_halfedge_handle(
                    self.mesh.next_halfedge_handle(
                        self.mesh.next_halfedge_handle(he)))
                if (not he.is_valid()) or he.idx() == he_idx or self.mesh.is_boundary(he):
                    break
        self.mesh.set_halfedge_property_array('theta', theta)

        Theta = np.zeros(num_vertices)
        k_theta = np.zeros(num_vertices)
        for vit in self.mesh.vertices():
            for he in self.mesh.voh(vit):
                if self.mesh.is_boundary(he):
                    continue
                he_prev = self.mesh.prev_halfedge_handle(he)
                Theta[vit.idx()] += self.mesh.calc_sector_angle(he_prev)

            if self.mesh.is_boundary(vit):
                # k_theta[vit.idx()] = pi / Theta[vit.idx()]
                k_theta[vit.idx()] = 2 * pi / Theta[vit.idx()]
                # k_theta[vit.idx()] = 1
            else:
                k_theta[vit.idx()] = 2 * pi / Theta[vit.idx()]
        self.mesh.set_vertex_property_array('Theta', Theta)
        self.mesh.set_vertex_property_array('k_theta', k_theta)
        return k_theta

    # for a vertex
    def _tangent2angle(self, tangent, v_idx, allowReverse=False):
        angle = 0
        xaxis = self.mesh.vertex_property_array('x_axis').astype(int)
        he = self.mesh.halfedge_handle(int(xaxis[v_idx]))
        notFound = False
        k = 1000 if allowReverse else 1
        while True:
            if self.mesh.face_handle(he).is_valid():
                he_prev = self.mesh.prev_halfedge_handle(he)
                A = self.mesh.point(self.mesh.vertex_handle(v_idx))
                B = self.mesh.point(self.mesh.to_vertex_handle(he))
                C = self.mesh.point(self.mesh.from_vertex_handle(he_prev))
                tf, _theta = pointInFan(A + k * tangent, A, B, C)
                if tf:
                    angle += _theta
                    break  # find it
                else:
                    angle += self.mesh.calc_sector_angle(he_prev)

            he = self.mesh.opposite_halfedge_handle(
                self.mesh.next_halfedge_handle(self.mesh.next_halfedge_handle(he)))
            if (not he.is_valid()) or he.idx() == xaxis[v_idx] or self.mesh.is_boundary(he):
                notFound = True
                break

        if allowReverse and notFound and \
                (self.mesh.is_boundary(self.mesh.vertex_handle(v_idx))):
            t = tangent * -1
            angle = 0
            he = self.mesh.halfedge_handle(int(xaxis[v_idx]))

            while True:
                if self.mesh.face_handle(he).is_valid():
                    he_prev = self.mesh.prev_halfedge_handle(he)
                    A = self.mesh.point(self.mesh.vertex_handle(v_idx))
                    B = self.mesh.point(self.mesh.to_vertex_handle(he))
                    C = self.mesh.point(self.mesh.from_vertex_handle(he_prev))
                    tf, _theta = pointInFan(A + k * t, A, B, C)
                    if tf:
                        angle += _theta
                        break  # find it
                    else:
                        angle += self.mesh.calc_sector_angle(he_prev)

                he = self.mesh.opposite_halfedge_handle(
                    self.mesh.next_halfedge_handle(self.mesh.next_halfedge_handle(he)))

                if (not he.is_valid()) or he.idx() == xaxis[v_idx] or self.mesh.is_boundary(he):
                    print('v-{} vector filed projection Not found'.format(v_idx))
                    angle = 0
                    break

        # real_angle = angle * self.mesh.vertex_property_array('k_theta')[v_idx]
        real_angle = angle

        return real_angle, self.v

    def generatePhi(self, tangents):
        num_points = self.mesh.n_vertices()
        assert tangents.shape[0] == num_points
        angle = np.zeros(num_points)
        v = np.zeros(num_points)

        # get arg and norm
        for v_idx in range(num_points):
            angle[v_idx], v[v_idx] = self._tangent2angle(tangents[v_idx], v_idx, allowReverse=False)

        # self.mesh.set_vertex_property_array('phi', 2 * angle + pi)
        self.mesh.set_vertex_property_array('v', v)
        return angle

    def getOmega(self, vectors):
        edgeLength = self.mesh.edge_property_array('length')

        Phi = self.generatePhi(vectors)
        ev_idx = self.mesh.ev_indices()
        edge_based_Phi = Phi[ev_idx]
        phiI = edge_based_Phi[:, 0]
        phiJ = edge_based_Phi[:, 1]

        theta = self.mesh.halfedge_property_array('theta')
        theta_reshaped = theta.reshape((-1, 2))
        thetaIJ = theta_reshaped[:, 0]
        thetaJI = theta_reshaped[:, 1]

        omega = 0.5 * edgeLength * \
                (self.v * np.cos(phiI - thetaIJ) +
                 self.v * np.cos(phiJ - thetaJI + pi))

        return torch.from_numpy(omega)

    def Phi2Omega(self, Phi):
        num_points = self.mesh.n_vertices()
        assert Phi.shape[0] == num_points
        ev_idx = self.mesh.ev_indices()
        edge_based_Phi = Phi[ev_idx]
        phiI = edge_based_Phi[:, 0]
        phiJ = edge_based_Phi[:, 1]

        theta_reshaped = self.thetaIJ.reshape((-1, 2))
        thetaIJ = theta_reshaped[:, 0]
        thetaJI = theta_reshaped[:, 1]

        omega = 0.5 * self.edgeLength * \
                (self.v * torch.cos((phiI - thetaIJ)) +
                 self.v * torch.cos((phiJ - thetaJI + pi)))

        return omega

    def Phi2OmegaNew(self, Phi):
        if self.sij is None:
            theta_reshaped = self.thetaIJ.reshape((-1, 2))
            thetaIJ = theta_reshaped[:, 0]
            thetaJI = theta_reshaped[:, 1] + pi
            dtheta = thetaJI - thetaIJ
            rij = torch.cos(dtheta) + complex(0, 1) * torch.sin(dtheta)
            ev_idx = self.mesh.ev_indices()
            edge_based_Phi = Phi[ev_idx]
            phiI = edge_based_Phi[:, 0]
            XI = torch.cos(phiI) + complex(0, 1) * torch.sin(phiI)
            phiJ = edge_based_Phi[:, 1]
            XJ = torch.cos(phiJ) + complex(0, 1) * torch.sin(phiJ)
            rijXI = rij * XI
            # rij_XI_XJ = torch.dot(rij*XI, XJ)
            self.sij = torch.sgn(rijXI.real * XJ.real + rijXI.imag * XJ.imag).detach().clone()

        num_points = self.mesh.n_vertices()
        assert Phi.shape[0] == num_points
        ev_idx = self.mesh.ev_indices()
        edge_based_Phi = Phi[ev_idx]
        phiI = edge_based_Phi[:, 0]
        phiJ = edge_based_Phi[:, 1]

        theta_reshaped = self.thetaIJ.reshape((-1, 2))
        thetaIJ = theta_reshaped[:, 0]
        thetaJI = theta_reshaped[:, 1]

        self.Phi = Phi

        omega = 0.5 * self.edgeLength * \
                (self.v * torch.cos((phiI - thetaIJ)) +
                 self.v * torch.cos((self.sij * phiJ - thetaJI + pi)))
        return omega

    def getTangent(self, phi):
        v_start_he = self.mesh.vertex_property_array('x_axis').astype(int)
        s = np.zeros((self.mesh.n_vertices(), 3))
        dir = np.zeros((self.mesh.n_vertices(), 3))

        for vit in self.mesh.vertices():
            _P = self.mesh.point(vit)
            N = self.mesh.calc_vertex_normal(vit)
            he = self.mesh.halfedge_handle(v_start_he[vit.idx()])
            X = np.copy(self.mesh.point(self.mesh.to_vertex_handle(he))) - _P
            X -= np.dot(X, N) * N
            X /= np.linalg.norm(X)
            JX = np.cross(N, X)
            _phi = phi[vit.idx()]
            s[vit.idx()] = _P
            dir[vit.idx()] = (cos(_phi) * X + sin(_phi) * JX)
        return s, dir

    def getTextureCoordiate(self, omega, psy):
        # omega = self.mesh.halfedge_property_array('omega')
        newOmega = np.zeros((omega.shape[0], 2))
        newOmega[:, 0] = omega
        newOmega[:, 1] = -omega
        omega = newOmega.reshape(-1)
        _j = complex(0, 1)
        RealImagePsi = psy.reshape(-1, 2)
        psi = RealImagePsi[:, 0] + _j * RealImagePsi[:, 1]
        paramIndex = np.zeros(self.mesh.n_faces())
        textureCoodient_u = np.zeros(self.mesh.n_halfedges())
        sij_he = torch.vstack((self.sij, self.sij)).T.flatten()

        self.mesh.request_halfedge_texcoords2D()
        alpha = []
        for fit in self.mesh.faces():
            heij = self.mesh.halfedge_handle(fit)
            if self.mesh.is_boundary(heij):
                continue

            heki = self.mesh.halfedge_handle(fit)
            heij = self.mesh.next_halfedge_handle(heki)
            hejk = self.mesh.next_halfedge_handle(heij)

            vI = self.mesh.from_vertex_handle(heij)
            vJ = self.mesh.from_vertex_handle(hejk)
            vK = self.mesh.from_vertex_handle(heki)

            psiI = psi[vI.idx()]
            psiJ = psi[vJ.idx()]
            psiK = psi[vK.idx()]

            # grab the connection coeffients
            omegaIJ = omega[heij.idx()]
            omegaJK = omega[hejk.idx()]
            omegaKI = omega[heki.idx()]

            crossesSheetsIJ = sij_he[heij.idx()].item()
            cIJ = 1 if self.mesh.halfedge_handle(self.mesh.edge_handle(heij), 0) == heij else -1
            cJK = 1 if self.mesh.halfedge_handle(self.mesh.edge_handle(hejk), 0) == hejk else -1
            cKI = 1 if self.mesh.halfedge_handle(self.mesh.edge_handle(heki), 0) == heki else -1

            psiJ_ = (1 + crossesSheetsIJ) / 2 * psiJ + (1 - crossesSheetsIJ) / 2 * psiJ.conj()
            omegaIJ = (1 + crossesSheetsIJ) / 2 * omegaIJ + (1 - crossesSheetsIJ) / 2 * cIJ * omegaIJ
            omegaJK = (1 + crossesSheetsIJ) / 2 * omegaJK + (1 - crossesSheetsIJ) / 2 * -cJK * omegaJK

            crossesSheetsKI = sij_he[heki.idx()].item()
            psiK_ = (1 + crossesSheetsKI) / 2 * psiK + (1 - crossesSheetsKI) / 2 * psiK.conj()
            omegaKI = (1 + crossesSheetsKI) / 2 * omegaKI + (1 - crossesSheetsKI) / 2 * -cKI * omegaKI
            omegaJK = (1 + crossesSheetsKI) / 2 * omegaJK + (1 - crossesSheetsKI) / 2 * cJK * omegaJK

            # construct complex transport coefficients
            rij = complex(cos(omegaIJ), sin(omegaIJ))
            rjk = complex(cos(omegaJK), sin(omegaJK))
            rki = complex(cos(omegaKI), sin(omegaKI))

            alphaI = arg(psiI)
            alphaJ = alphaI + omegaIJ - arg(rij * psiI / psiJ_)
            alphaK = alphaJ + omegaJK - arg(rjk * psiJ_ / psiK_)
            alphaL = alphaK + omegaKI - arg(rki * psiK_ / psiI)

            n = np.round((alphaL - alphaI) / (2. * pi))
            # alphaJ -= 2. * pi * n / 3.
            # alphaK -= 4. * pi * n / 3.

            textureCoodient_u[heij.idx()] = alphaI
            textureCoodient_u[hejk.idx()] = alphaJ
            textureCoodient_u[heki.idx()] = alphaK
            paramIndex[fit.idx()] = n

            self.mesh.set_texcoord2D(heki, [alphaI / (2. * pi) + 0.5, 0.5])
            self.mesh.set_texcoord2D(heij, [alphaJ / (2. * pi) + 0.5, 0.5])
            self.mesh.set_texcoord2D(hejk, [alphaK / (2. * pi) + 0.5, 0.5])

            alpha.append([alphaJ, alphaK, alphaI])

        self.mesh.set_halfedge_property_array('tu', textureCoodient_u)
        self.mesh.set_face_property_array('paramIndex', paramIndex)

    def getTextureCoordiate_th(self, omega, psi):
        newOmega = torch.zeros((omega.shape[0], 2))
        newOmega[:, 0] = omega
        newOmega[:, 1] = -omega
        omega = newOmega.reshape(-1)
        _j = complex(0, 1)
        RealImagePsi = psi.reshape(-1, 2)
        Psi = torch.view_as_complex(RealImagePsi)

        nb_face_idx = self.non_boundary_faces_idx
        # fv = torch.tensor(self.mesh.fv_indices()[nb_face_idx], dtype=torch.long)
        # fh = torch.tensor(self.mesh.fh_indices()[nb_face_idx], dtype=torch.long)

        fv = torch.tensor(self.mesh.fv_indices(), dtype=torch.long)
        fh = torch.tensor(self.mesh.fh_indices(), dtype=torch.long)

        self.area = torch.from_numpy(self.generateArea()).float()
        # area = self.area[nb_face_idx]
        area = self.area

        # heIsConic = self.mesh.eh_indices()

        fvI, fvJ, fvK = fv.unbind(-1)
        fhKI, fhIJ, fhJK = fh.unbind(-1)
        cKI, cIJ, cJK = -2 * (fhKI % 2 - 0.5), -2 * (fhIJ % 2 - 0.5), -2 * (fhJK % 2 - 0.5)

        psiI, psiJ, psiK = Psi[fvI], Psi[fvJ], Psi[fvK]
        omegaIJ, omegaJK, omegaKI = omega[fhIJ], omega[fhJK], omega[fhKI]

        sij_he = torch.vstack((self.sij, self.sij)).T.flatten()
        crossesSheetsIJ = sij_he[fhIJ]
        psiJ_ = (1 + crossesSheetsIJ) / 2 * psiJ + (1 - crossesSheetsIJ) / 2 * psiJ.conj()
        omegaIJ = (1 + crossesSheetsIJ) / 2 * omegaIJ + (1 - crossesSheetsIJ) / 2 * cIJ * omegaIJ
        omegaJK = (1 + crossesSheetsIJ) / 2 * omegaJK + (1 - crossesSheetsIJ) / 2 * -cJK * omegaJK

        crossesSheetsKI = sij_he[fhKI]
        psiK_ = (1 + crossesSheetsKI) / 2 * psiK + (1 - crossesSheetsKI) / 2 * psiK.conj()
        omegaKI = (1 + crossesSheetsKI) / 2 * omegaKI + (1 - crossesSheetsKI) / 2 * -cKI * omegaKI
        omegaJK = (1 + crossesSheetsKI) / 2 * omegaJK + (1 - crossesSheetsKI) / 2 * cJK * omegaJK

        rIJ = torch.cos(omegaIJ) + torch.sin(omegaIJ) * _j
        rJK = torch.cos(omegaJK) + torch.sin(omegaJK) * _j
        rKI = torch.cos(omegaKI) + torch.sin(omegaKI) * _j

        alphaI = psiI.angle()
        alphaJ = alphaI + omegaIJ - (rIJ * psiI / (psiJ_ + 1e-9)).angle()
        alphaK = alphaJ + omegaJK - (rJK * psiJ_ / (psiK_ + 1e-9)).angle()
        alphaL = alphaK + omegaKI - (rKI * psiK_ / (psiI + 1e-9)).angle()

        # n = torch.round((alphaL - alphaI) / (2. * pi))
        n = ((alphaL - alphaI) / (2. * pi))
        alphaI_ = alphaI
        alphaJ_ = alphaJ - 2. * pi * n / 3.
        alphaK_ = alphaK - 4. * pi * n / 3.
        alpha = torch.vstack((alphaI_, alphaJ_, alphaK_))

        # self.paramIndex[nb_face_idx] = n
        self.paramIndex = n

        vI, vJ, vK = self.points[fvI], self.points[fvJ], self.points[fvK]
        edgeIJ = vJ - vI
        edgeJK = vK - vJ
        edgeKI = vI - vK

        deltaIJ = alphaJ_ - alphaI_
        deltaJK = alphaK_ - alphaJ_
        deltaKI = alphaI_ - alphaK_

        grad = (edgeIJ * deltaIJ.unsqueeze(-1).expand(-1, 3) +
                edgeJK * deltaJK.unsqueeze(-1).expand(-1, 3) +
                edgeKI * deltaKI.unsqueeze(-1).expand(-1, 3)) / \
               (3 * area.unsqueeze(-1).expand(-1, 3))
        grad_ = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-6)
        return grad_, alpha, n

    # not use, ignored
    def getTextureCoordiate_boundary_th(self, omega, psi):
        newOmega = torch.zeros((omega.shape[0], 2))
        newOmega[:, 0] = omega
        newOmega[:, 1] = -omega
        _j = complex(0, 1)
        RealImagePsi = psi.reshape(-1, 2)
        Psi = torch.view_as_complex(RealImagePsi) + 1e-5

        b_face_idx = self.boundary_faces_idx
        fv = torch.tensor(self.mesh.fv_indices()[b_face_idx], dtype=torch.long)
        self.area = torch.from_numpy(self.generateArea()).float()
        area = self.area[b_face_idx]

        fvI, fvJ, fvK = fv.unbind(-1)
        psiI, psiJ, psiK = Psi[fvI], Psi[fvJ], Psi[fvK]

        vI, vJ, vK = self.points[fvI], self.points[fvJ], self.points[fvK]
        edgeIJ = vJ - vI
        edgeJK = vK - vJ
        edgeKI = vI - vK

        alphaI = psiI.angle()
        alphaJ = psiJ.angle()
        alphaK = psiK.angle()
        alpha = torch.vstack((alphaI, alphaJ, alphaK))

        deltaIJ = alphaJ - alphaI
        deltaJK = alphaK - alphaJ
        deltaKI = alphaI - alphaK

        self.paramIndex[b_face_idx] = torch.zeros(b_face_idx.shape[0])

        grad = (edgeIJ * deltaIJ.unsqueeze(-1).expand(-1, 3) +
                edgeJK * deltaJK.unsqueeze(-1).expand(-1, 3) +
                edgeKI * deltaKI.unsqueeze(-1).expand(-1, 3)) / \
               (3 * area.unsqueeze(-1).expand(-1, 3))
        grad_ = grad / (torch.linalg.norm(grad, dim=1, keepdim=True) + 1e-6)
        return grad_, alpha

    def fieldIndex(self, fidx, Phi, thetaIJ, k=2):
        def fmodPi(theta):
            return theta - (2. * pi) * floor((theta + pi) / (2. * pi))

        fit = self.mesh.face_handle(fidx)
        Omega = 0
        index = 0
        if self.mesh.is_boundary(fit):
            return 0
        start_he = self.mesh.halfedge_handle(fit)
        he = start_he
        while True:
            I = self.mesh.from_vertex_handle(he)
            J = self.mesh.to_vertex_handle(he)
            phiI, phiJ = fmodPI(Phi[I.idx()]) * k + pi, fmodPI(Phi[J.idx()]) * k + pi
            thetaI, thetaJ = thetaIJ[I.idx()], thetaIJ[J.idx()] + pi
            dTheta = thetaI - thetaJ
            Omega += dTheta
            index += fmodPi(phiJ - phiI + k * dTheta)
            he = self.mesh.next_halfedge_handle(he)

            if he == start_he:
                break

        index -= k * fmodPi(Omega)
        return index

    def glueParameterization(self):
        num_faces = self.mesh.n_faces()

        _j = complex(0, 1)
        tu = self.mesh.halfedge_property_array('tu')
        tv = np.zeros(self.mesh.n_halfedges()) + 0.5
        paramIndex = self.mesh.face_property_array('paramIndex')
        # paramIndex[self.boundary_faces_idx] = 0

        Q = []
        visited = np.zeros((num_faces))
        visited[self.ignoredFlipFaces] = 1

        f0 = None
        for fit in self.mesh.faces():
            if abs(paramIndex[fit.idx()]) > 1e-5:
                continue
            if not self.mesh.is_boundary(fit):
                f0 = fit
                break

        if f0 is None:
            print('Error: Cannot find a valid face to fix!')

        visited[f0.idx()] = True
        Q.append(f0)

        while len(Q) > 0:
            f = Q.pop(0)
            # print(f.idx())
            he = self.mesh.halfedge_handle(f)
            while True:
                fj = self.mesh.face_handle(self.mesh.opposite_halfedge_handle(he))
                # indexJ = self.fieldIndex(fj.idx(), self.Phi, self.thetaIJ)
                '''
                if self.mesh.is_valid_handle(fj) and\
                        not self.fieldIndex(fj.idx(), self.Phi, self.thetaIJ)==0:
                    visited[fj.idx()] = 1
                # print(fj.idx())
                # if (not self.mesh.is_boundary(he)) and (not visited[fj.idx()]) and paramIndex[fj.idx()] == 0:
                '''
                if (not self.mesh.is_boundary(he)) and \
                        (not visited[fj.idx()]) and \
                        paramIndex[fj.idx()] == 0:
                    he_next = self.mesh.next_halfedge_handle(he)
                    he_oppo = self.mesh.opposite_halfedge_handle(he)
                    he_oppo_next = self.mesh.next_halfedge_handle(he_oppo)
                    he_oppo_next_next = self.mesh.next_halfedge_handle(he_oppo_next)

                    bi = tu[he_oppo_next.idx()] + tv[he_oppo_next.idx()] * _j
                    bj = tu[he_oppo.idx()] + tv[he_oppo.idx()] * _j
                    bk = tu[he_oppo_next_next.idx()] + tv[he_oppo_next_next.idx()] * _j

                    ai = tu[he.idx()] + tv[he.idx()] * _j
                    aj = tu[he_next.idx()] + tv[he_next.idx()] * _j

                    u = aj - ai
                    v = bj - bi

                    theta = arg(u / (v + 1e-7))
                    z = complex(cos(theta), sin(theta))
                    if abs(z.imag) > 1e-9:
                        bi = bi.conjugate()
                        bj = bj.conjugate()
                        bk = bk.conjugate()

                    v = bj - bi

                    theta = arg(u / (v + 1e-7))
                    z = complex(cos(theta), sin(theta))

                    if abs(z.imag) < 1e-5:
                        b0 = bi
                        bi = z * (bi - b0) + ai
                        bj = z * (bj - b0) + ai
                        bk = z * (bk - b0) + ai

                    tu[he_oppo_next.idx()] = bi.real
                    tu[he_oppo.idx()] = bj.real
                    tu[he_oppo_next_next.idx()] = bk.real

                    self.mesh.set_texcoord2D(he_oppo, np.array([bi.real / (pi * 2) + 0.5, 0.5]))
                    self.mesh.set_texcoord2D(he_oppo_next_next, np.array([bj.real / (pi * 2) + 0.5, 0.5]))
                    self.mesh.set_texcoord2D(he_oppo_next, np.array([bk.real / (pi * 2) + 0.5, 0.5]))

                    # self.mesh.set_texcoord2D(he_oppo, np.array([bi.real, 0.5]))
                    # self.mesh.set_texcoord2D(he_oppo_next_next, np.array([bj.real, 0.5]))
                    # self.mesh.set_texcoord2D(he_oppo_next, np.array([bk.real, 0.5]))

                    visited[fj.idx()] = True
                    if paramIndex[fj.idx()] == 0:
                        Q.append(fj)

                he = self.mesh.next_halfedge_handle(he)
                if he == self.mesh.halfedge_handle(f):
                    break

        self.visited = np.array(visited)

        # openmesh.write_mesh('out.obj', self.mesh, halfedge_tex_coord=True)

        vtu = np.zeros((self.mesh.n_faces() * 3, 2), dtype=float)
        for fit in self.mesh.faces():
            heij = self.mesh.halfedge_handle(fit)
            vtu[fit.idx() * 3 + 0] = self.mesh.texcoord2D(heij)
            heij = self.mesh.next_halfedge_handle(heij)
            vtu[fit.idx() * 3 + 1] = self.mesh.texcoord2D(heij)
            heij = self.mesh.next_halfedge_handle(heij)
            vtu[fit.idx() * 3 + 2] = self.mesh.texcoord2D(heij)
        return visited, vtu

    def energy(self, omega, Psi):
        ev_idx = self.mesh.ev_indices()
        _j = complex(0, 1)

        RealImagePsi = Psi.reshape(-1, 2)
        ComplexPsi = RealImagePsi[:, 0] + RealImagePsi[:, 1] * _j
        w = self.weight
        complexE = (ComplexPsi[ev_idx[:, 1]] - np.exp(_j * omega) * ComplexPsi[ev_idx[:, 0]])
        e = (complexE * torch.conj(complexE) * w)
        # assert (sum(abs(e)) - Psi.T@self.A@Psi)<1e-5
        return e

    def postprocess(self, vt, savepath='', draw=False):
        faces = self.pvmesh.faces.reshape(-1, 4)[:, 1:]
        points = self.pvmesh.points[faces.flatten()]
        mesh_om = self.mesh

        boundary_faces = []
        for vit in mesh_om.vertices():
            if mesh_om.is_boundary(vit):
                for vfit in mesh_om.vf(vit):
                    boundary_faces.append(vfit.idx())
        boundary_faces_idx = np.array(boundary_faces)

        # vt = self.mesh.halfedge_property_array('tu')

        mesh_with_repeted_points = pv.make_tri_mesh(points, np.arange(faces.shape[0] * 3).reshape(-1, 3))

        paramIndex = self.mesh.face_property_array('paramIndex')
        mesh_with_repeted_points.active_texture_coordinates = vt

        # stress_mesh = drawVectorEachFace(self.mesh, stress_np, 0.01)

        stripe_values = vt[:, 0].reshape(-1, 3)
        stripe_values *= 2

        # extract iso-lines from non-Singularity regions
        newParamIndex = paramIndex.copy()
        line_points, line_points_distance, line_edges, line_indicesPerEdge, edgeIndicesofFaces = \
            self.extractCrossingsFromStripePattern(stripe_values, newParamIndex, np.zeros(self.pvmesh.n_cells),
                                                   boundary_faces_idx)
        # connect iso-lines in Singularity regions
        self.connectIsolinesOnSingularitiesNew(newParamIndex, line_points, line_points_distance, line_edges, line_indicesPerEdge,
                                               edgeIndicesofFaces, boundary_faces_idx)

        lines = [[len(e)] + e for e in line_edges]
        stripe_iso_line = pv.PolyData(line_points, lines=lines)
        # stripe_iso_line.point_data['boundary_distance'] = line_points_distance

        if savepath is None or len(savepath) == 0:
            # savePath2ObjExt('outpath.obj', stripe_iso_line, edgeIndicesofFaces)
            pass
        else:
            savePath2ObjExt(savepath, stripe_iso_line, edgeIndicesofFaces)

        # draw = True
        if draw:
            self.texture = pv.read_texture(r'data/t_b1.png')

            # grad_mesh = drawGrad(mesh, psi, grad)
            # grad_mesh = drawVectorEachFace(self.mesh, grad, 0.01)
            singularity_mesh_idx = np.where(abs(paramIndex) > 0.5)[0]
            mesh_with_repeted_points.compute_normals(point_normals=True, cell_normals=False, inplace=True)
            singularity_mesh = mesh_with_repeted_points.extract_cells(singularity_mesh_idx)
            offset_distance = 1e-2
            new_points = singularity_mesh.points + offset_distance * singularity_mesh.active_normals
            singularity_mesh.points = new_points

            plotter = pv.Plotter()
            plotter.add_axes()
            plotter.add_mesh(mesh_with_repeted_points, texture=self.texture, show_edges=True)
            plotter.add_mesh(singularity_mesh, color='k', label='singularity')

            # plotter.add_mesh(grad_mesh, color='g', label='gradient')
            # plotter.add_mesh(stress_mesh, color='r', label='stress')

            plotter.add_mesh(stripe_iso_line, color='r', label='iso_lines', show_edges=True)
            plotter.show()

        return stripe_iso_line, edgeIndicesofFaces

    # find all iso-line in regular region, i.e. nparam == 0
    def extractCrossingsFromStripePattern(self, stripeValues, stripeIndices, fieldIndices, boundary_faces_idx):
        # list of per-edge indices pointing to the vertices list
        mesh = self.pvmesh
        if 'boundary_distance' in mesh.point_data:
            boundary_distance = mesh.point_data['boundary_distance']
        else:
            boundary_distance = np.zeros(mesh.points.shape[0])

        faces = mesh.faces.reshape(-1, 4)[:, 1:]

        isBoundaryFaces = np.zeros(mesh.n_cells)
        isBoundaryFaces[boundary_faces_idx] = 1

        # points = mesh.points[faces.flatten()]
        mesh_om = openmesh.TriMesh(mesh.points, faces)
        polylineIndices = [[] for _ in range(mesh_om.n_edges())]
        vertices = []
        vertices_distance = []

        edges = []
        edgeIndicesofFaces = []

        for fit in mesh_om.faces():
            # singularities are ignored in this function
            edgeIndices = [[], [], []]
            if stripeIndices[fit.idx()] != 0 or \
                    fieldIndices[fit.idx()] != 0 or \
                    isBoundaryFaces[fit.idx()]:
                continue
            fv = mesh_om.fv_indices()[fit.idx()]
            fs = stripeValues[fit.idx()]
            fe = mesh_om.fe_indices()[fit.idx()]
            for i in range(3):
                v0, v1 = fv[i], fv[(i + 1) % 3]
                s0, s1 = fs[i], fs[(i + 1) % 3]
                e0 = fe[(i + 1) % 3]

                isoPoints = list(range(ceil(min(s0, s1)), floor(max(s0, s1)) + 1))
                isoPoints = [(ips - min(s0, s1)) / abs(s1 - s0) for ips in isoPoints]
                if s0 > s1:
                    isoPoints = [1 - ips for ips in isoPoints]
                    isoPoints.reverse()

                if len(polylineIndices[e0]) == 0:
                    for bary in isoPoints:
                        polylineIndices[e0].append(len(vertices))
                        vertices.append(bary * mesh_om.points()[v1] + (1 - bary) * mesh_om.points()[v0])
                        vertices_distance.append(bary * boundary_distance[v1] + (1 - bary) * boundary_distance[v0])

                    edgeIndices[i] = polylineIndices[e0].copy()
                else:
                    pi_e0_reverse = polylineIndices[e0].copy()
                    pi_e0_reverse.reverse()
                    edgeIndices[i] += pi_e0_reverse

            matching = []
            try:
                matching = self.matchCrossings(edgeIndices)
            except Exception as e:
                # print(str(e))
                pass

            # matching = matchCrossings(edgeIndices)

            edges += matching
            edgeIndicesofFaces += [fit.idx() for _ in matching]
        return vertices, vertices_distance, edges, polylineIndices, edgeIndicesofFaces

    # connect point pair via geo-path
    def connectGeoDistanceInMesh(self, facesIdx, line_vertices, point_pair):
        if len(facesIdx) == 1:
            if point_pair[2] == 2:
                f = self.pvmesh.faces.reshape((-1, 4))[facesIdx[0], 1:]
                face_center = self.pvmesh.points[f].mean(axis=0)
                return np.vstack([line_vertices[point_pair[0]], face_center, line_vertices[point_pair[1]]])
            else:
                return np.vstack([line_vertices[point_pair[0]], line_vertices[point_pair[1]]])

        self.pvmesh.cell_data['ori_idx'] = np.arange(self.pvmesh.n_cells)
        sub_mesh = self.pvmesh.extract_cells(facesIdx)
        sub_mesh_f = sub_mesh.cells.reshape(-1, 4)[:, 1:]
        sub_mesh_om = openmesh.TriMesh(sub_mesh.points, sub_mesh_f)

        def pointInEdge(mesh_om, eit, point):
            heit = mesh_om.halfedge_handle(eit, 0)
            from_vh = mesh_om.from_vertex_handle(heit)
            to_vh = mesh_om.to_vertex_handle(heit)
            point1 = mesh_om.point(from_vh)
            point2 = mesh_om.point(to_vh)
            p1 = np.array(point1)
            p2 = np.array(point2)
            p = np.array(point)
            edge_vector = p2 - p1
            point_vector = p - p1
            cross_product = np.cross(edge_vector, point_vector)
            if np.linalg.norm(cross_product) > 1e-5:
                return None
            dot_product = np.dot(point_vector, edge_vector)
            if dot_product < 0 or dot_product > np.dot(edge_vector, edge_vector):
                return None
            newV = mesh_om.add_vertex(point)
            mesh_om.split(mesh_om.edge_handle(heit), newV)
            return newV

        P0 = line_vertices[point_pair[0]]
        P1 = line_vertices[point_pair[1]]
        v = [-1, -1]

        for eit in sub_mesh_om.edges():
            if not sub_mesh_om.is_boundary(eit):
                continue
            newV0 = pointInEdge(sub_mesh_om, eit, P0)
            if newV0 is not None:
                v[0] = newV0.idx()

        for eit in sub_mesh_om.edges():
            if not sub_mesh_om.is_boundary(eit):
                continue
            newV1 = pointInEdge(sub_mesh_om, eit, P1)
            if newV1 is not None:
                v[1] = newV1.idx()

        if v[0] == -1 or v[1] == -1:
            return np.vstack([line_vertices[point_pair[0]], line_vertices[point_pair[1]]])

        solver = pp3d.EdgeFlipGeodesicSolver(sub_mesh_om.points(), sub_mesh_om.fv_indices())
        path_pts = solver.find_geodesic_path(v_start=v[0], v_end=v[1])
        return path_pts

    # connect edges in singularities region
    def connectIsolinesOnSingularitiesNew(self, paramIndex, line_vertices, line_vertices_distance,
                                          line_edges, indicesPerEdge,
                                          edgeIndicesofFaces, boundary_faces_idx):
        v2e = np.zeros(len(line_vertices), dtype=int)

        isBoundaryFaces = np.zeros(self.pvmesh.faces.shape[0])
        isBoundaryFaces[boundary_faces_idx] = 1

        for eid, eit in enumerate(indicesPerEdge):
            for evit in eit:
                v2e[evit] = eid

        # find all connected singulartity regions
        singularity_mesh_idx = np.where(abs(paramIndex) > 0.5)[0]
        visited = np.zeros(self.pvmesh.n_cells) + 1
        visited[singularity_mesh_idx] = 0
        visited_num_faces = 0
        all_connected_regions = []

        fe = self.mesh.fe_indices()
        ef = self.mesh.ef_indices()
        fv = self.mesh.fv_indices()

        for sfitid in singularity_mesh_idx:
            if isBoundaryFaces[sfitid]:
                continue

            need_to_connect = []
            if visited_num_faces == singularity_mesh_idx.shape[0]:
                break
            if visited[sfitid]:
                continue
            all_connected_faces = [sfitid]
            old_all_connected_faces_num = 1

            # get all irregular connected faces
            while True:
                sfit = self.mesh.face_handle(sfitid)
                for ffit in self.mesh.ff(sfit):
                    if (not visited[ffit.idx()]) and \
                            ffit.idx() not in all_connected_faces and \
                            (not isBoundaryFaces[ffit.idx()]):
                        visited[ffit.idx()] = 1
                        all_connected_faces.append(ffit.idx())
                if old_all_connected_faces_num == len(all_connected_faces):
                    break
                old_all_connected_faces_num = len(all_connected_faces)

            all_edges = fe[all_connected_faces]
            all_edges_unique, counts = np.unique(all_edges.flatten(), return_counts=True)
            boundary_edges = all_edges_unique[np.argwhere(counts == 1)].flatten()

            all_bounary_points = [p for e in boundary_edges for p in indicesPerEdge[e]]
            all_faces_paramIndex = [paramIndex[f] for f in all_connected_faces]

            for acf in all_connected_faces:
                paramIndex[acf] = 0

            # connect all boundary vertices
            if len(all_bounary_points) < 2:
                for acf in all_connected_faces:
                    paramIndex[acf] = 0

            elif len(all_bounary_points) == 2:
                all_bounary_points.append(2)
                need_to_connect.append(all_bounary_points)
                for acf in all_connected_faces:
                    paramIndex[acf] = 0

            # via Hungarian Algorithm
            else:
                lp = np.asarray([line_vertices[p] for p in all_bounary_points])
                dis_matrix = scipy.spatial.distance.cdist(lp, lp)
                dis_matrix += np.diag(dis_matrix.sum(1))

                for idp1, p1 in enumerate(all_bounary_points):
                    for idp2, p2 in enumerate(all_bounary_points):
                        # similar same edge 2, same face 1
                        similar = len(np.intersect1d(ef[v2e[p1]], ef[v2e[p2]]))
                        dis_matrix[idp1][idp2] += similar

                row_ind, col_ind = scipy.optimize.linear_sum_assignment(dis_matrix)
                initial_assignments = list(zip(row_ind, col_ind))

                # check unique
                used = set()
                final_assignments = []

                for r, c in initial_assignments:
                    if r not in used and c not in used and r < c:
                        final_assignments.append((r, c))
                        used.add(r)
                        used.add(c)

                filtered_pairs = [[all_bounary_points[i], all_bounary_points[j], len(all_bounary_points)]
                                  for i, j in final_assignments]
                need_to_connect += filtered_pairs

            for pair in need_to_connect:
                newPath = self.connectGeoDistanceInMesh(all_connected_faces, line_vertices, pair)
                v = [0] * len(newPath)
                v[0], v[-1] = pair[0], pair[1]
                if len(newPath) == 2:
                    line_edges.append(pair[:2])
                    lineinFanIdx = -1
                    for f in all_connected_faces:
                        P = (line_vertices[pair[0]] + line_vertices[pair[1]]) / 2
                        tf, _ = pointInFan(P, self.pvmesh.points[fv[f, 0]],
                                           self.pvmesh.points[fv[f, 1]],
                                           self.pvmesh.points[fv[f, 2]])
                        if tf:
                            lineinFanIdx = f
                    edgeIndicesofFaces.append(lineinFanIdx)
                    assert lineinFanIdx != -1
                else:
                    for i in range(1, len(newPath) - 1):
                        line_vertices.append(newPath[i])
                        v[i] = len(line_vertices) - 1
                    for i in range(len(newPath) - 1):
                        line_edges.append([v[i], v[i + 1]])
                        lineinFanIdx = -1
                        for f in all_connected_faces:
                            P = (line_vertices[v[i]] + line_vertices[v[i + 1]]) / 2
                            tf, _ = pointInFan(P, self.pvmesh.points[fv[f, 0]],
                                               self.pvmesh.points[fv[f, 1]],
                                               self.pvmesh.points[fv[f, 2]])
                            if tf:
                                lineinFanIdx = f
                        edgeIndicesofFaces.append(lineinFanIdx)
                        assert lineinFanIdx != -1

            all_connected_regions.append(all_connected_faces)

    # Matches crossings based on a strategy proposed in "Navigating intrinsic triangulations" [Sharp et al. 2019].
    # See https://github.com/nmwsharp/geometry-central/pull/89#issuecomment-936150222 for more details
    def matchCrossings(self, crossings):
        assert len(crossings) == 3
        idxIJ = 2
        if (len(crossings[0]) >= len(crossings[1]) and len(crossings[0]) >= len(crossings[2])):
            idxIJ = 0
        elif len(crossings[1]) >= len(crossings[2]) and len(crossings[1]) >= len(crossings[0]):
            idxIJ = 1

        idxJK = (idxIJ + 1) % 3
        idxKI = (idxIJ + 2) % 3

        IJ = crossings[idxIJ]
        JK = crossings[idxJK]
        KI = crossings[idxKI]

        nIJ = len(IJ)
        nJK = len(JK)
        nKI = len(KI)

        assert (nIJ >= nJK and nIJ >= nKI)
        assert (nIJ <= nJK + nKI)
        assert ((nIJ + nJK + nKI) % 2 == 0)

        matchings = []
        if (nIJ == nJK + nKI):
            # Case 1: all edges intersecting ijk cross a common edge ij
            # match IJ with IK
            for m in range(nKI):
                matchings.append([IJ[m], KI[nKI - m - 1]])
            # match IJ with KJ
            for m in range(nJK):
                matchings.append([IJ[nKI + m], JK[nJK - m - 1]])
        else:
            # Case 2: there is no common edge
            nRemainingCrossings = (nIJ + nJK + nKI) / 2
            m = 0
            while (nRemainingCrossings > nJK):
                matchings.append([IJ[m], KI[nKI - m - 1]])
                m += 1
                nRemainingCrossings -= 1

            l = 0
            while (nRemainingCrossings > nKI - m):
                matchings.append([IJ[nIJ - 1 - l], JK[l]])
                nRemainingCrossings -= 1
                l += 1

            p = 0
            while (nRemainingCrossings > 0):
                matchings.append([JK[nJK - 1 - p], KI[p]])
                p += 1
                nRemainingCrossings -= 1
        return matchings
