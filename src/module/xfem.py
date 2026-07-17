import numpy as np
import torch
from .FE import FE

class XFEM:
    def __init__(self, problem, device='cuda'):
        self.device = device
        self.mesh = problem.mesh
        self.material = problem.materialProperty
        
        # Initialize material properties for two isotropic materials
        self.material1 = {
            'E': self.material.get('E1', 1.0),  # Young's modulus for material 1
            'nu': self.material.get('nu1', 0.3),  # Poisson's ratio for material 1
            'penal': self.material.get('penal', 3.0)  # Penalization factor
        }
        self.material2 = {
            'E': self.material.get('E2', 0.1),  # Young's modulus for material 2
            'nu': self.material.get('nu2', 0.3),  # Poisson's ratio for material 2
            'penal': self.material.get('penal', 3.0)  # Penalization factor
        }
        
        # Initialize standard FE solver for reference
        self.fe = FE(problem, device=device)
        
        # Initialize enrichment parameters
        self.enrichment_radius = 2.0  # Radius around interface for enrichment
        self.enriched_nodes = None
        self.enriched_elements = None
        
    def get_enrichment_functions(self, phi):
        """Calculate enrichment functions using |φ(x)|"""
        # Get absolute value of level set function
        abs_phi = torch.abs(phi)
        
        # Identify elements cut by the interface (where phi changes sign)
        cut_elements = torch.where(torch.prod(phi[self.mesh.elemNodes], dim=1) <= 0)[0]
        
        # Find nodes to be enriched (within enrichment radius of interface)
        node_distances = torch.zeros(self.mesh.numNodes, device=self.device)
        for elem in cut_elements:
            elem_nodes = self.mesh.elemNodes[elem]
            node_distances[elem_nodes] = torch.min(node_distances[elem_nodes], 
                                                 abs_phi[elem_nodes])
        
        # Nodes to be enriched are those within enrichment radius
        self.enriched_nodes = torch.where(node_distances <= self.enrichment_radius)[0]
        
        # Elements containing enriched nodes
        self.enriched_elements = torch.unique(torch.where(
            torch.isin(self.mesh.elemNodes, self.enriched_nodes))[0])
        
        # Calculate enrichment functions
        N = torch.zeros((self.mesh.numNodes, len(self.enriched_nodes)), device=self.device)
        for i, node in enumerate(self.enriched_nodes):
            # Heaviside enrichment function
            N[node, i] = torch.sign(phi[node])
            
            # Get connected elements
            connected_elems = torch.where(torch.isin(self.mesh.elemNodes, node))[0]
            
            # For elements containing enriched node, calculate |φ(x)|
            for elem in connected_elems:
                elem_nodes = self.mesh.elemNodes[elem]
                N[elem_nodes, i] = torch.abs(phi[elem_nodes])
        
        return N
    
    def get_element_stiffness(self, elem, phi, density):
        """Calculate element stiffness matrix for isotropic material"""
        # Get element nodes
        elem_nodes = self.mesh.elemNodes[elem]
        
        # Determine which material to use based on level set
        phi_elem = phi[elem_nodes].mean()  # Average phi value for element
        
        if phi_elem > 0:
            E = self.material1['E']
            nu = self.material1['nu']
        else:
            E = self.material2['E']
            nu = self.material2['nu']
            
        # Get element stiffness matrix from standard FE
        ke = torch.tensor(self.mesh.KE[elem], device=self.device)
        
        # Apply material properties and density
        ke = ke * E * (1e-3 + density[elem]) ** self.material1['penal']
        
        return ke
    
    def assemble_stiffness_matrix(self, phi, density):
        """Assemble the enriched stiffness matrix for isotropic materials"""
        # Get enrichment functions
        N = self.get_enrichment_functions(phi)
        
        # Assemble global stiffness matrix with enrichment
        n_dof = self.mesh.ndof
        n_enriched = len(self.enriched_nodes)
        total_dof = n_dof + n_enriched
        
        # Initialize sparse matrix indices and values
        iK = []
        jK = []
        sK = []
        
        # Standard DOF contributions
        for elem in range(self.mesh.numElems):
            dofs = self.mesh.edofMat[elem]
            ke = self.get_element_stiffness(elem, phi, density)
            
            # Add standard DOF contributions
            for i in range(len(dofs)):
                for j in range(len(dofs)):
                    iK.append(dofs[i])
                    jK.append(dofs[j])
                    sK.append(ke[i,j])
            
            # Add enriched DOF contributions if element is enriched
            if elem in self.enriched_elements:
                elem_nodes = self.mesh.elemNodes[elem]
                enriched_dofs = torch.where(torch.isin(self.enriched_nodes, elem_nodes))[0]
                
                for i in range(len(dofs)):
                    for j in range(len(enriched_dofs)):
                        # Standard-enriched coupling
                        iK.extend([dofs[i], n_dof + enriched_dofs[j]])
                        jK.extend([n_dof + enriched_dofs[j], dofs[i]])
                        sK.extend([ke[i,j] * N[elem_nodes[j], enriched_dofs[j]],
                                 ke[i,j] * N[elem_nodes[j], enriched_dofs[j]]])
                
                for i in range(len(enriched_dofs)):
                    for j in range(len(enriched_dofs)):
                        # Enriched-enriched coupling
                        iK.append(n_dof + enriched_dofs[i])
                        jK.append(n_dof + enriched_dofs[j])
                        sK.append(ke[i,j] * N[elem_nodes[i], enriched_dofs[i]] * 
                                N[elem_nodes[j], enriched_dofs[j]])
        
        # Convert to sparse matrix
        K = torch.sparse_coo_tensor(
            indices=torch.stack([torch.tensor(iK, device=self.device),
                               torch.tensor(jK, device=self.device)]),
            values=torch.tensor(sK, device=self.device),
            size=(total_dof, total_dof)
        )
        
        return K
    
    def solve(self, phi, density):
        """Solve the XFEM system for isotropic materials"""
        # Assemble stiffness matrix
        K = self.assemble_stiffness_matrix(phi, density)
        
        # Get enrichment functions
        N = self.get_enrichment_functions(phi)
        n_dof = self.mesh.ndof
        n_enriched = len(self.enriched_nodes)
        
        # Assemble force vector with enrichment
        f = torch.zeros(n_dof + n_enriched, device=self.device)
        f[:n_dof] = self.mesh.f[:, 0]  # Standard forces
        
        # Add enriched forces
        for i, node in enumerate(self.enriched_nodes):
            f[n_dof + i] = self.mesh.f[node, 0] * N[node, i]
        
        # Apply boundary conditions
        free_dofs = torch.setdiff1d(torch.arange(n_dof + n_enriched, device=self.device),
                                  torch.tensor(self.mesh.fixed, device=self.device))
        
        # Solve system
        K_free = K[free_dofs][:, free_dofs]
        f_free = f[free_dofs]
        
        # Use sparse solver
        u_free = torch.linalg.solve(K_free.to_dense(), f_free)
        
        # Reconstruct full solution
        u = torch.zeros(n_dof + n_enriched, device=self.device)
        u[free_dofs] = u_free
        
        # Calculate compliance
        compliance = torch.dot(u, f)
        
        return u, compliance 