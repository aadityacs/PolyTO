import numpy as np
import jax.numpy as jnp
import jax
from mesher import Mesher
from typing import Tuple
import dataclasses
import matplotlib.pyplot as plt

@dataclasses.dataclass
class BC:
  force: np.ndarray
  fixed_dofs: np.ndarray

  @property
  def num_dofs(self):
    return self.force.shape[0]

  @property
  def free_dofs(self):
    return jnp.setdiff1d(np.arange(self.num_dofs),self.fixed_dofs)

@dataclasses.dataclass
class Material:
  E: float = 1.
  nu: float = 0.3
  rho_min: float = 1e-3

#-------------------------#

class FEA:
  def __init__(self, mesh: Mesher, material: Material, bc: BC):
    self.mesh, self.material, self.bc = mesh, material, bc
    self.dofs_per_elem, self.num_dofs  = 8, 2*mesh.num_nodes
    self.D0 = self.FE_compute_element_stiffness()
    self.elem_node, self.edofMat, self.iK, self.jK = \
        self.compute_connectivity_info()
  #-----------------------#
  def FE_compute_element_stiffness(self) -> np.ndarray:
    E = self.material.E
    nu = self.material.nu
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,\
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    D0 = E/(1-nu**2)*np.array([\
    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]).T
    return D0
  #-----------------------#
  def compute_connectivity_info(self) \
      -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    elem_node = np.zeros((4, self.mesh.num_elems))
    for elx in range(self.mesh.nelx):
      for ely in range(self.mesh.nely):
        el = ely+elx*self.mesh.nely
        n1=(self.mesh.nely+1)*elx+ely
        n2=(self.mesh.nely+1)*(elx+1)+ely
        elem_node[:,el] = np.array([n1+1, n2+1, n2, n1])
    elem_node = elem_node.astype(int)

    edofMat = np.zeros( ( self.mesh.num_elems ,\
        4*self.mesh.num_dim ), dtype=int )

    for elem in range(self.mesh.num_elems):
      enodes = elem_node[:,elem]

      edofs = np.stack( ( 2*enodes , 2*enodes+1 ) , axis=1 )\
              .reshape( ( 1 , self.dofs_per_elem ) )
      edofMat[elem,:] = edofs
    
    matrx_size = self.mesh.num_elems*self.dofs_per_elem**2
    iK = np.kron( edofMat , np.ones((self.dofs_per_elem,1),dtype=int) ).T\
        .reshape( matrx_size, order ='F')
    jK = np.kron( edofMat , np.ones((1,self.dofs_per_elem),dtype=int) ).T\
        .reshape( matrx_size , order ='F')
    return elem_node, edofMat, iK, jK
  #---------------#
  def compute_elem_stiffness_matrix(self, density: jnp.ndarray, 
                                    penal = 3., rho_min = 1e-2)->jnp.ndarray:
    """
    Args:
      density: Array of size (num_elems,) which is the density of each of the
        element. The entries are in [0,1] where 0 means the element is void
        and 1 means the element is filled with material.
        penal: SIMP penalty parameter
        rho_min: A small value added to the density to ensure that the values are
          slightly greater than zero. This is done to ensure numerical stability
          during the simulation
    Returns: Array of size (num_elems, 8, 8) which is the structual
      stiffness matrix of each of the bilinear quad elements. Each element has
      8 dofs corresponding to the x and y displacements of the 4 noded quad
      element.
    """
    penalized_dens =  rho_min + density**penal
    # e - element, i - elem_nodes j - elem_nodes
    return jnp.einsum('e, ij->ije', penalized_dens, self.D0)
  #-----------------------#
  def assemble_stiffness_matrix(self, elem_stiff_mtrx: jnp.ndarray):
    """
    Args:
      elem_stiff_mtrx: Array of size (num_elems, 8, 8) which is the structual
        stiffness matrix of each of the bilinear quad elements. Each element has
        8 dofs corresponding to the x and y displacements of the 4 noded quad
        element.
    Returns: Array of size (num_dofs, num_dofs) which is the assembled global
      stiffness matrix.
    """
    glob_stiff_mtrx = jnp.zeros((self.num_dofs, self.num_dofs))
    glob_stiff_mtrx = glob_stiff_mtrx.at[(self.iK, self.jK)].add(
                                      elem_stiff_mtrx.flatten('F'))
    return glob_stiff_mtrx
  #-----------------------#
  def solve(self, glob_stiff_mtrx):
    """Solve the system of Finite element equations.
    Args:
      glob_stiff_mtrx: Array of size (num_dofs, num_dofs) which is the assembled
        global stiffness matrix.
    Returns: Array of size (num_dofs,) which is the displacement of the nodes.
    """
    k_free = glob_stiff_mtrx[self.bc.free_dofs,:][:,self.bc.free_dofs]

    u_free = jax.scipy.linalg.solve(
          k_free,
          self.bc.force[self.bc.free_dofs], \
          sym_pos = True, check_finite=False)
    u = jnp.zeros((self.num_dofs))
    u = u.at[self.bc.free_dofs].add(u_free.reshape(-1))
    return u
  #-----------------------#
  def compute_compliance(self, u:jnp.ndarray)->jnp.ndarray:
    """Objective measure for structural performance.
    Args:
      u: Array of size (num_dofs,) which is the displacement of the nodes
        of the mesh.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    return jnp.dot(u, self.bc.force.flatten() )
  #-----------------------#
  def loss_function(self, density:jnp.ndarray)->float:
    """Wrapper function that takes in density field and returns compliance.
    Args:
      density: Array of size (num_elems,) that contain the density of each
        of the elements for FEA.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    elem_stiffness_mtrx = self.compute_elem_stiffness_matrix(density)
    glob_stiff_mtrx = self.assemble_stiffness_matrix(elem_stiffness_mtrx)
    u = self.solve(glob_stiff_mtrx)
    return self.compute_compliance(u)
  #-----------------------#
  def plot_displacement(self, u, density = None):
    elemDisp = u[self.edofMat].reshape(self.mesh.nelx*self.mesh.nely, 8)
    elemU = (elemDisp[:,0] + elemDisp[:,2] + elemDisp[:,4] + elemDisp[:,6])/4
    elemV = (elemDisp[:,1] + elemDisp[:,3] + elemDisp[:,5] + elemDisp[:,7])/4
    delta = np.sqrt(elemU**2 + elemV**2)
    x, y = np.mgrid[:self.mesh.nelx, :self.mesh.nely]
    scale = 0.1*max(self.mesh.nelx, self.mesh.nely)/max(delta)
    x = x + scale*elemU.reshape(self.mesh.nelx, self.mesh.nely)
    y = y + scale*elemV.reshape(self.mesh.nelx, self.mesh.nely)

    if density is not None:
      delta = delta*np.round(density)
    z = delta.reshape(self.mesh.nelx, self.mesh.nely)
    im = plt.pcolormesh(x, y, z, cmap='coolwarm')
    plt.title('deformation')
    plt.colorbar(im)