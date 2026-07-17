import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# ---------------- Geometry ----------------
nodes = np.array([
    [0.0, 2.0], [0.0, 1.5], [0.0, 1.0], [0.0, 0.0],
    [1.0, 2.0], [1.0, 1.5], [1.0, 1.0], [1.0, 0.0],
])
elements = [
    [0, 4, 5, 1],  # elem 0
    [1, 5, 6, 2],  # elem 1 (cut)
    [2, 6, 7, 3],  # elem 2
]

def phi(x, y):
    return y + 0.2*x - 1.25

cut_elem = 1
cut_nodes = elements[cut_elem]
enriched_nodes = sorted(set(cut_nodes))
enr_map = {n:i for i,n in enumerate(enriched_nodes)}

n_nodes = len(nodes)
std_dofs = 2*n_nodes
total_dofs = std_dofs + 2*len(enriched_nodes)

# -------------- Materials ---------------
def D_mat(E,nu):
    fac = E/(1-nu**2)
    return fac*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
D_soft = D_mat(1e6,0.35)
D_hard = D_mat(2e9,0.25)

# -------------- Shape Q4 ---------------
def shape_Q4(xi,eta):
    N = 0.25*np.array([(1-xi)*(1-eta),
                       (1+xi)*(1-eta),
                       (1+xi)*(1+eta),
                       (1-xi)*(1+eta)])
    dN_dxi = 0.25*np.array([[-(1-eta),-(1-xi)],
                             [+(1-eta),-(1+xi)],
                             [+(1+eta), +(1+xi)],
                             [-(1+eta), +(1-xi)]])
    return N,dN_dxi

gauss=[(-np.sqrt(1/3),-np.sqrt(1/3)),
       ( np.sqrt(1/3),-np.sqrt(1/3)),
       ( np.sqrt(1/3), np.sqrt(1/3)),
       (-np.sqrt(1/3), np.sqrt(1/3))]

# -------------- Assembly ----------------
K = lil_matrix((total_dofs,total_dofs))
f = np.zeros(total_dofs)

for eid,conn in enumerate(elements):
    coords = nodes[conn]
    is_cut = (eid==cut_elem)
    local_enr_flags = [n in enriched_nodes for n in conn]
    nenr = sum(local_enr_flags) if is_cut else 0
    ndof_local = 8 + 2*nenr
    Ke = np.zeros((ndof_local, ndof_local))
    
    for xi,eta in gauss:
        N, dNdxi = shape_Q4(xi,eta)
        J = np.zeros((2,2))
        for a in range(4):
            J += np.outer(dNdxi[a], coords[a])
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dNdx = np.array([invJ@dNdxi[a] for a in range(4)])
        
        B_std = np.zeros((3,8))
        for a in range(4):
            B_std[0,2*a]   = dNdx[a,0]
            B_std[1,2*a+1] = dNdx[a,1]
            B_std[2,2*a]   = dNdx[a,1]
            B_std[2,2*a+1] = dNdx[a,0]
        
        xgp = N@coords[:,0]; ygp = N@coords[:,1]
        D = D_soft if phi(xgp,ygp)>0 else D_hard
        
        if is_cut:
            phi_gp = phi(xgp,ygp)
            psi = abs(phi_gp)
            sign = np.sign(phi_gp) if phi_gp!=0 else 0
            grad_phi = np.array([0.2,1.0])
            grad_psi = sign*grad_phi
            
            B_enr = np.zeros((3,2*nenr))
            idx=0
            for a in range(4):
                if local_enr_flags[a]:
                    grad_Npsi = psi*dNdx[a] + N[a]*grad_psi
                    B_enr[0,2*idx]   = grad_Npsi[0]
                    B_enr[1,2*idx+1] = grad_Npsi[1]
                    B_enr[2,2*idx]   = grad_Npsi[1]
                    B_enr[2,2*idx+1] = grad_Npsi[0]
                    idx+=1
            B = np.hstack([B_std, B_enr])
        else:
            B = B_std
        
        Ke += B.T @ D @ B * detJ
    
    # Global DOF mapping
    gdofs=[]
    for n in conn:
        gdofs.extend([2*n,2*n+1])
    if is_cut:
        for n in conn:
            if n in enriched_nodes:
                base = std_dofs+2*enr_map[n]
                gdofs.extend([base,base+1])
    # assemble
    for i,a in enumerate(gdofs):
        for j,b in enumerate(gdofs):
            K[a,b]+=Ke[i,j]

# -------------- Loads & BC --------------
f[2*0+1]=1e3
f[2*4+1]=1e3
fixed=[3,7]
fixed_dofs=[]
for n in fixed:
    fixed_dofs.extend([2*n,2*n+1])
free=np.setdiff1d(np.arange(total_dofs), fixed_dofs)

K_ff = K[free[:,None], free].tocsc()
u = np.zeros(total_dofs)
u[free] = spsolve(K_ff, f[free])
disp=u[:std_dofs].reshape(-1,2)

print("Node : ux (m) , uy (m)")
for i,(ux,uy) in enumerate(disp):
    print(f"{i:2d}: {ux:.4e} , {uy:.4e}")

# -------------- Plot -----------------
scale=2e2
def_nodes = nodes + scale*disp
fig,ax=plt.subplots(figsize=(4,6))
for elem in elements:
    poly = nodes[elem+[elem[0]]]
    ax.plot(poly[:,0], poly[:,1], 'k--')
    poly2 = def_nodes[elem+[elem[0]]]
    ax.plot(poly2[:,0], poly2[:,1], 'r-')
ax.set_aspect('equal')
ax.set_title('Deformed (red) vs original (black)')
plt.tight_layout()
plt.show()
