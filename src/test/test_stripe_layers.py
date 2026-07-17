# from utils.tetIO import *
import tetgen
import numpy as np
import pyvista as pv
import scipy
import scipy.sparse as ssp
# Try different import patterns
try:
    from src.utils.tetIO import loadTet
except ImportError:
    try:
        from utils.tetIO import loadTet
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.tetIO import loadTet

def printingLayerVectorField(xyz: np.array):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    X = np.vstack((0 * x, - y - 5 / 3, z + 0.1)).T
    # X = np.vstack((0 * x, 0 * x, 0 * x + 1)).T
    X += 1e-9
    return X


def getAllEdges(elem):
    edges = set()
    for idx, it in enumerate(elem):
        vit = it.copy()
        vit.sort()
        e1 = '{}_{}'.format(vit[0], vit[1])
        e2 = '{}_{}'.format(vit[0], vit[2])
        e3 = '{}_{}'.format(vit[0], vit[3])
        e4 = '{}_{}'.format(vit[1], vit[2])
        e5 = '{}_{}'.format(vit[1], vit[3])
        e6 = '{}_{}'.format(vit[2], vit[3])

        edges.add(e1)
        edges.add(e2)
        edges.add(e3)
        edges.add(e4)
        edges.add(e5)
        edges.add(e6)

    # edgesList = list(edges)
    edges = [[int(eit.split('_')[0]), int(eit.split('_')[1])] for eit in edges]
    return np.array(edges, dtype=int)


def getAllFaces(elem):
    faces = set()
    for idx, it in enumerate(elem):
        vit = it.copy()
        vit.sort()
        f1 = '{}_{}_{}'.format(vit[0], vit[1], vit[2])
        f2 = '{}_{}_{}'.format(vit[0], vit[2], vit[3])
        f3 = '{}_{}_{}'.format(vit[0], vit[3], vit[1])
        f4 = '{}_{}_{}'.format(vit[1], vit[3], vit[2])

        faces.add(f1)
        faces.add(f2)
        faces.add(f3)
        faces.add(f4)

    faces = [[int(fit.split('_')[0]), int(fit.split('_')[1]), int(fit.split('_')[2])] for fit in faces]
    return np.array(faces, dtype=int)


# face 2 edge and face 2 half-edge
def getFE(face, edge):
    fe = list()
    fh = list()

    edges = dict()
    for eidx, eit in enumerate(edge):
        e = '{}_{}'.format(eit[0], eit[1])
        edges[e] = eidx

    for fidx, fit in enumerate(face):
        fit_copy = fit.copy()

        e1 = '{}_{}'.format(min(fit_copy[0], fit_copy[1]), max(fit_copy[0], fit_copy[1]))
        e2 = '{}_{}'.format(min(fit_copy[1], fit_copy[2]), max(fit_copy[1], fit_copy[2]))
        e3 = '{}_{}'.format(min(fit_copy[0], fit_copy[2]), max(fit_copy[0], fit_copy[2]))

        fh1 = 1 if fit[0] > fit[1] else 0
        fh2 = 1 if fit[1] > fit[2] else 0
        fh3 = 1 if fit[2] > fit[0] else 0

        fe.append([edges[e1], edges[e2], edges[e3]])
        fh.append([edges[e1] * 2 + fh1, edges[e2] * 2 + fh2, edges[e3] * 2 + fh3])

    return np.array(fe), np.array(fh)


def generateTexture(omega, psy, node, elem, faces, edges):
    newOmega = np.zeros((omega.shape[0], 2))
    newOmega[:, 0] = omega
    newOmega[:, 1] = -omega
    omega = newOmega.reshape(-1)
    _j = complex(0, 1)
    RealImagePsi = psy.reshape(-1, 2)
    RealImagePsi /= np.linalg.norm(RealImagePsi, axis=1, keepdims=True)
    Psi = RealImagePsi.view(dtype=np.complex128)[..., 0]

    # fv = torch.tensor(self.mesh.fv_indices()[nb_face_idx], dtype=torch.long)
    # fh = torch.tensor(self.mesh.fh_indices()[nb_face_idx], dtype=torch.long)

    fv = faces
    _, fh = getFE(faces, edges)

    area = 1

    fvI, fvJ, fvK = fv[:, 0], fv[:, 1], fv[:, 2]
    fhIJ, fhJK, fhKI = fh[:, 0], fh[:, 1], fh[:, 2]
    cKI, cIJ, cJK = -2 * (fhKI % 2 - 0.5), -2 * (fhIJ % 2 - 0.5), -2 * (fhJK % 2 - 0.5)

    psiI, psiJ, psiK = Psi[fvI], Psi[fvJ], Psi[fvK]
    omegaIJ, omegaJK, omegaKI = omega[fhIJ], omega[fhJK], omega[fhKI]

    '''
    crossesSheetsIJ = 1
    psiJ_ = (1 + crossesSheetsIJ) / 2 * psiJ + (1 - crossesSheetsIJ) / 2 * psiJ.conj()
    omegaIJ = (1 + crossesSheetsIJ) / 2 * omegaIJ + (1 - crossesSheetsIJ) / 2 * cIJ * omegaIJ
    omegaJK = (1 + crossesSheetsIJ) / 2 * omegaJK + (1 - crossesSheetsIJ) / 2 * -cJK * omegaJK

    crossesSheetsKI = 1
    psiK_ = (1 + crossesSheetsKI) / 2 * psiK 
    omegaKI = (1 + crossesSheetsKI) / 2 * omegaKI + (1 - crossesSheetsKI) / 2 * -cKI * omegaKI
    omegaJK = (1 + crossesSheetsKI) / 2 * omegaJK + (1 - crossesSheetsKI) / 2 * cJK * omegaJK
    '''

    psiJ_ = psiJ
    psiK_ = psiK
    #
    rIJ = np.cos(omegaIJ) + np.sin(omegaIJ) * _j
    rJK = np.cos(omegaJK) + np.sin(omegaJK) * _j
    rKI = np.cos(omegaKI) + np.sin(omegaKI) * _j

    alphaI = np.angle(psiI)
    alphaJ = alphaI + omegaIJ - np.angle(rIJ * psiI / (psiJ_ + 1e-9))
    alphaK = alphaJ + omegaJK - np.angle(rJK * psiJ_ / (psiK_ + 1e-9))
    alphaL = alphaK + omegaKI - np.angle(rKI * psiK_ / (psiI + 1e-9))

    # n = torch.round((alphaL - alphaI) / (2. * math.pi))
    n = ((alphaL - alphaI) / (2. * np.pi))
    alphaI_ = alphaI
    alphaJ_ = alphaJ - 2. * np.pi * n / 3.
    alphaK_ = alphaK - 4. * np.pi * n / 3.
    alpha = np.vstack((alphaI_, alphaJ_, alphaK_))
    paramIndex = n


    '''
    vI, vJ, vK = node[fvI], node[fvJ], node[fvK]
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
    grad_ = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-6)
    '''
    return None, alpha, paramIndex


if __name__ == "__main__":
    import os

    os.environ["DISPLAY"] = "localhost:10.0"
    os.environ["HOME"] = "/home/ryan/"

    v = 20

    meshPath = r'src/test/Wedge2.obj'
    # meshPath = r'src/test/spot_triangulated.obj'
    mesh = pv.read(meshPath)
    mesh.points *= 10
    # mesh = pv.Cube(center=(0, 0, 0), x_length=10, y_length=10, z_length=10)
 
    v = 5

    sourceOrgMeshTet = tetgen.TetGen(mesh)
    # sourceOrgMeshTet.make_manifold(verbose=True)
    # newV, newF = sourceOrgMeshTet.v.copy(), sourceOrgMeshTet.f.copy()
    # node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.5/60a{}Y".format(100), verbose=verbose)
    node, elem = sourceOrgMeshTet.tetrahedralize(verbose=True)
    grid = sourceOrgMeshTet.grid
    
    '''
    grid = loadTet(r'src/test/topopt_new4.tet')
    grid.points -= grid.points.min(axis=0)
    grid.points /= grid.points.max()
    grid.points *= 10

    
    # print(grid.cells.shape)
    elem = grid.cells.reshape(-1, 5)[:, 1:]
    node = grid.points
    '''
    
    allEdges = getAllEdges(elem)
    allFaces = getAllFaces(elem)
    fe, fh = getFE(allFaces, allEdges)

    X = printingLayerVectorField(node)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    ve = node[allEdges]
    e_ij = ve[:, 0] - ve[:, 1]
    l_ij = np.linalg.norm(e_ij, axis=1)

    Xe = X[allEdges]
    a_ij = 0.5 * ((e_ij * Xe[:, 0, :]).sum(1) + (e_ij * Xe[:, 1, :]).sum(1))

    omega = v * a_ij

    # scalar field gradient
    '''
    A = np.zeros((allEdges.shape[0] + 1, node.shape[0] + 1))
    for idx, eit in enumerate(allEdges):
        A[idx, eit[0]] = 1
        A[idx, eit[1]] = -1
    A[-1, 0] = 1
    A[0, -1] = 1
    '''

    X_b = np.hstack((omega, np.array(0)))

    # complex scalar field gradient
    weight = 1
    cosOmega = np.cos(omega) * weight
    sinOmega = np.sin(omega) * weight
    vW = np.hstack((-cosOmega, -sinOmega, sinOmega, -cosOmega,
                    -cosOmega, sinOmega, -sinOmega, -cosOmega)).flatten()
    # vW = torch.hstack((-cosOmega, -sinOmega, sinOmega, -cosOmega,
    #                    -cosOmega, sinOmega, -sinOmega, -cosOmega)).flatten()
    nRange = np.arange(2 * node.shape[0])

    vs, ve = allEdges[:, 0], allEdges[:, 1]
    I, J = 2 * vs, 2 * ve
    neRange = np.arange(0, allEdges.shape[0])

    iW = np.hstack([I, I, I + 1, I + 1, J, J, J + 1, J + 1]).flatten()
    jW = np.hstack([J, J + 1, J, J + 1, I, I + 1, I, I + 1]).flatten()

    _A = ssp.coo_matrix((vW, (iW, jW)), shape=[2 * node.shape[0] + 2, 2 * node.shape[0] + 2])
    nonzero_per_row = np.bincount(_A.row) / 2

    # diagA = _A.sum(0)
    diagA = np.concatenate((nonzero_per_row, np.array([0, 0])))
    A = _A + ssp.diags(diagA + 1e-4)
    A = A.todense()
    
    np.savez('out.npz', A=A, vW=vW, iW=iW, jW=jW)

    A[-1, 0] = 1
    A[-2, 1] = 1
    A[0, -1] = 1
    A[1, -2] = 1

    X_b = np.zeros((2 * node.shape[0] + 2))
    X_b[-1] = 1

    lstq_res = np.linalg.lstsq(A, X_b)
    psy = lstq_res[0]
    psy_complex = psy.reshape(-1, 2).view(dtype=np.complex128)[..., 0]
    arg = np.angle(psy_complex)

    grad_, alpha, paramIndex = generateTexture(omega, psy, node, elem, allFaces, allEdges)

    grid.point_data['alpha'] = arg[:-1]
    grid.point_data['vector_field'] = X

    texture = pv.read_texture(r'src/test/t1.png')
    new_points = node[allFaces.flatten()]
    mesh_with_repeted_points = pv.make_tri_mesh(new_points, np.arange(allFaces.shape[0] * 3).reshape(-1, 3))
    uv_t = np.vstack((alpha.T.flatten()/np.pi, np.zeros_like(alpha.flatten())+0.5)).T
    mesh_with_repeted_points.active_texture_coordinates = uv_t
    vectorFieldGlyphs = grid.glyph(scale=False, orient='vector_field', factor=0.4)

    plotter = pv.Plotter()
    # plotter.add_mesh(mesh_with_repeted_points, texture=texture)
    plotter.add_mesh(grid, style='wireframe', cmap='hsv')
    plotter.add_mesh(vectorFieldGlyphs)
    plotter.add_axes()
    plotter.show()

    mesh_with_repeted_points.save('out1.obj')
    vectorFieldGlyphs.save('glyphs.vtk')
