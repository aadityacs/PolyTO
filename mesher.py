import numpy as np
from dataclasses import dataclass

@dataclass
class BoundingBox:
  """Coordinates of the south-west and north-east points of the bounding box."""
  x_min: float
  x_max: float
  y_min: float
  y_max: float


class Mesher:
  """
  Attributes:
    num_dim: number of dimensions of the mesh. Currently only handles 2D
    nelx: number of elements along X axis
    nely: number of elements along Y axis
    num_elems: number of elements in the mesh
    bounding_box: Contains the max and min coordinates of the mesh
    lx: length of domain along X axis
    ly: length of domain along Y axis
    elem_size: Array which contains the size of the element along X and Y axis
    elem_area: Area of each element
    domain_volume: Area of the rectangular domain
    num_nodes: number of nodes in the mesh. Assume a bilinear quad element
    elem_centers: Array of size (num_elems, 2) which are the coordinates of the
      centers of the element
  """
  def __init__(self, nelx:int, nely:int, bounding_box: BoundingBox):
    self.num_dim = 2
    self.nelx, self.nely = nelx, nely
    self.num_elems = nelx*nely
    self.bounding_box = bounding_box
    self.lx = np.abs(self.bounding_box.x_max - self.bounding_box.x_min)
    self.ly = np.abs(self.bounding_box.y_max - self.bounding_box.y_min)
    dx, dy = self.lx/nelx, self.ly/nely
    self.elem_size = np.array([dx, dy])
    self.elem_area = dx*dy
    self.domain_volume = self.elem_area*self.num_elems
    self.num_nodes = (nelx+1)*(nely+1)

    [x_grid, y_grid] = np.meshgrid(
               np.linspace(self.bounding_box.x_min + dx/2.,
                           self.bounding_box.x_max-dx/2.,
                           nelx),
               np.linspace(self.bounding_box.y_min + dy/2.,
                           self.bounding_box.y_max-dy/2.,
                           nely))
    self.elem_centers = np.stack((x_grid, y_grid)).T.reshape(-1, self.num_dim)