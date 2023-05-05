from enum import Enum, auto
import numpy as np
import mesher
import FE_routines

_Mesher = mesher.Mesher
_BC = FE_routines.BC


class SampleBoundaryConditions(Enum):
  MID_CANT_BEAM = auto()
  MBB_BEAM = auto()

def get_sample_bc(mesh:_Mesher, sample:SampleBoundaryConditions):
  if(sample == SampleBoundaryConditions.MID_CANT_BEAM):
    force = np.zeros((2*mesh.num_nodes,1))
    dofs=np.arange(2*mesh.num_nodes)
    fixed = dofs[0:2*(mesh.nely+1):1]
    force[2*(mesh.nelx+1)*(mesh.nely+1)- 1*(mesh.nely+1), 0 ] = -1.
  if(sample == SampleBoundaryConditions.MBB_BEAM):
    force = np.zeros((2*mesh.num_nodes,1))
    dofs=np.arange(2*mesh.num_nodes)
    fixed= np.union1d(np.arange(0,2*(mesh.nely+1),2), 
                      2*(mesh.nelx+1)*(mesh.nely+1)-2*(mesh.nely+1)+1)
    force[2*(mesh.nely+1)+1 ,0]= -1.

  return _BC(force=force, fixed_dofs=fixed)