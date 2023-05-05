import jax.numpy as jnp
import jax
import mesher
import numpy as np
import dataclasses
from enum import Enum, auto
_mesh = mesher.Mesher

@dataclasses.dataclass
class PolygonExtents:
  """Hyper-parameters of the size, number and configurations of the polygons.
  Attributes:
    num_polys: number of polygons that occupy the design domain.
    num_planes_in_a_poly: number of planes that define each polygon.
    min_center_x: smallest x center coordinate the polys can have.
    min_center_y: smallest y center coordinate the polys can have.
    max_center_x: largest x center coordinate the polys can have.
    max_center_y: largest y center coordinate the polys can have.
    min_face_offset: smallest offset of a plane from the center.
    max_face_offset: largest offset of a plane from the center.
    min_angle_offset: smallest angular offset of a polygon.
    max_angle_offset: largest angular offset of a polygon.
  """
  num_polys: int
  num_planes_in_a_poly: int
  min_center_x: float
  min_center_y: float
  max_center_x: float
  max_center_y: float
  min_face_offset: float
  max_face_offset: float
  min_angle_offset: float
  max_angle_offset: float

@dataclasses.dataclass
class ConvexPolys:
  """A dataclass to describe convex polys.
  
  Each poly is described by a set of hyperplanes. The polys are inspired from:

  Deng, B., etal (2020). CVXNet: Learnable convex decomposition. CVPR, 31–41.
  https://doi.org/10.1109/CVPR42600.2020.00011

  However to ensure that the hyperplanes always form a bounded region the normals
  are fixed and is decided by the number of planes (N) in each poly. We assume the
  normal angle of the first plane is 0, (2pi)/N in the second, and i(2pi)/N of
  the ith plane. Each plane is further offset by a prescribed distance from the
  center of each poly.

  Attributes:
    center_x: Array of size (num_polys,) with the x-center of the polys.
    center_y: Array of size (num_polys,) with the y-center of the polys.
    angle_offset: Array of size (num_polys,) of the angle offset of the polys in
      radians.
    face_offset: Array of size (num_polys, num_planes_in_a_polys) with the 
      offset of each of the planes of the faces.
  """
  center_x: jnp.ndarray
  center_y: jnp.ndarray
  angle_offset: jnp.ndarray
  face_offset: jnp.ndarray

  def __post_init__(self):
    if(self.num_planes_in_a_poly < 3):
      raise Exception(f"Expect atleast 3 sides, got {self.num_planes_in_a_poly}")

  @property
  def num_free_parameters(self):
    """Number of free parameters for optimization."""
    return self.num_polys*(self.num_planes_in_a_poly + 3)

  @property
  def face_angles(self):
    """ Array of size (num_polys, num_planes_in_poly) """
    return (self.angle_offset[:, np.newaxis] + 
             jnp.linspace(0, 2*np.pi, self.num_planes_in_a_poly + 1)[:-1])

  @property
  def num_polys(self):
    return self.face_offset.shape[0]

  @property
  def num_planes_in_a_poly(self):
    return self.face_offset.shape[1]

  @classmethod
  def from_array(
      cls,
      state_array: jnp.ndarray,
      num_polys: int,
      num_planes_in_a_poly: int,
  ) -> 'ConvexPolys':
    """Converts a rank-1 array into `ConvexPolys`."""
    cx = state_array[0:num_polys]
    cy = state_array[num_polys:2*num_polys]
    ang = state_array[2*num_polys:3*num_polys]
    offset = state_array[3*num_polys:].reshape((num_polys, num_planes_in_a_poly))

    return ConvexPolys(cx, cy, ang, offset)

  def to_array(self) -> np.ndarray:
    """Converts the `ConvexPolys` into a rank-1 array."""
    return jnp.concatenate([f.reshape((-1)) for f in dataclasses.astuple(self)])

  def to_normalized_array(self, poly_extents: PolygonExtents) -> jnp.ndarray:
    """Converts the `ConvexPolys` into a rank-1 array with values normalized."""
    range_cx = (poly_extents.max_center_x - poly_extents.min_center_x)
    cx = (self.center_x - poly_extents.min_center_x)/range_cx

    range_cy = (poly_extents.max_center_y - poly_extents.min_center_y)
    cy = (self.center_y - poly_extents.min_center_y)/range_cy

    range_ang = (poly_extents.max_angle_offset - poly_extents.min_angle_offset)
    ang = (self.angle_offset - poly_extents.min_angle_offset)/range_ang

    range_offset = (poly_extents.max_face_offset - 
               poly_extents.min_face_offset)
    offset = (self.face_offset - poly_extents.min_face_offset)/range_offset

    return jnp.concatenate(( cx.reshape((-1)), cy.reshape((-1)), 
                            ang.reshape((-1)), offset.reshape((-1)) ))

  @classmethod
  def from_normalized_array(cls, state_array: jnp.ndarray,
      poly_extents: PolygonExtents)->'ConvexPolys':
    """Converts a normalized rank-1 array into `ConvexPolys`."""
    nb = poly_extents.num_polys
    np = poly_extents.num_planes_in_a_poly

    range_cx = (poly_extents.max_center_x - poly_extents.min_center_x)
    cx = state_array[0:nb]*range_cx + poly_extents.min_center_x

    range_cy = (poly_extents.max_center_y - poly_extents.min_center_y)
    cy = state_array[nb:2*nb]*range_cy + poly_extents.min_center_y

    range_ang = (poly_extents.max_angle_offset - poly_extents.min_angle_offset)
    ang = state_array[2*nb:3*nb]*range_ang + poly_extents.min_angle_offset

    range_offset = (poly_extents.max_face_offset - poly_extents.min_face_offset)
    
    offset = (state_array[3*nb:]*range_offset + poly_extents.min_face_offset
              ).reshape((nb, np))

    return ConvexPolys(cx, cy, ang, offset)

def compute_edge_lengths_of_polygons(polygons: ConvexPolys):
  """Compute the edge lengths of the polygon from the parameterization.

  For detailed derivation see: https://tinyurl.com/y4z5w4nu
  We have the distances of the hyperplanes from the center and the angles
  between adjacent lines. We can use this information to compute the length
  of the edges. A negative value of the length indicates that the line is
  farther away that it doesn't account as a side of the polygon also. This
  is useful in imposing minimum feature size constraints.

  Args:
    polygons: A set of polygons from dataclass `ConvexPolys`
  returns: An array of size (num_polys*num_planes_in_a_poly) that is the
    edge length of each side of the polygon.
  """
  alpha = 2*np.pi/polygons.num_planes_in_a_poly
  d_next = jnp.roll(polygons.face_offset, -1, axis=1)
  d_prev = jnp.roll(polygons.face_offset, 1, axis=1)

  edge_lengths = ((d_next + d_prev - 2*polygons.face_offset*jnp.cos(alpha))/
                  jnp.sin(alpha))
  
  return edge_lengths

def init_poly_grid(nx: int, ny: int, poly_extents: PolygonExtents):
  """
  NOTE: User ensures that the number of polys in `poly_extents` is set as
  `nx*ny`.

  Args:
   nx: number of polys along the x-axis.
   ny: number of polys along the y-axis.
   poly_extents: dataclass of `PolygonExtents` that contain metadata about the
    polys.
  
  Returns: A set of `nx*ny` equi-spaced and equi-sized `ConvexPolys`.
  """
  len_x = np.abs(poly_extents.max_center_x - poly_extents.min_center_x)
  len_y = np.abs(poly_extents.max_center_y - poly_extents.min_center_y)
  del_x = len_x/(4*nx)
  del_y = len_y/(4*ny)
  face_offset = min(del_x, del_y)*np.ones(
      (poly_extents.num_polys, poly_extents.num_planes_in_a_poly))
  cx = poly_extents.min_center_x + np.linspace(2*del_x, len_x - 2*del_x, nx)
  cy = poly_extents.min_center_y + np.linspace(2*del_y, len_y - 2*del_y, ny)
  [cx_grid,cy_grid] = np.meshgrid(cx, cy)
  mean_ang = 0.5*(poly_extents.max_angle_offset + poly_extents.min_angle_offset)
  ang_offset = mean_ang*np.ones((poly_extents.num_polys))
  return ConvexPolys(cx_grid.reshape((-1)), cy_grid.reshape((-1)), 
                    ang_offset, face_offset)


def init_random_polys(poly_extents: PolygonExtents, seed: int = 27):
  """Initialize the polys randomly.

  Args:
    poly_extents: dataclass of `PolygonExtents` that contain metadata about the
    polys.
    seed: Random seed to be used to ensure reproducibility.
  Returns: A set of randomly initialized `ConvexPolys`.
  """
  key = jax.random.PRNGKey(seed)
  cxkey, cykey, angkey, offkey = jax.random.split(key, 4)
  cx = jax.random.uniform(cxkey, (poly_extents.num_polys,),
            minval=poly_extents.min_center_x, maxval=poly_extents.max_center_x)
  cy = jax.random.uniform(cykey, (poly_extents.num_polys,),
            minval=poly_extents.min_center_y, maxval=poly_extents.max_center_y)
  ang = jax.random.uniform(angkey, (poly_extents.num_polys,),
            minval=poly_extents.min_angle_offset, maxval=poly_extents.max_angle_offset)
  off = jax.random.uniform(offkey, (poly_extents.num_polys, poly_extents.num_planes_in_a_poly),
            minval=poly_extents.min_face_offset, maxval=poly_extents.max_face_offset)
  mean_offset = 0.5*(poly_extents.min_face_offset + poly_extents.max_face_offset)
  off = mean_offset*jnp.ones((poly_extents.num_polys, poly_extents.num_planes_in_a_poly))
  return ConvexPolys(cx, cy, ang, off)

def compute_poly_sdf(polys: ConvexPolys, mesh: _mesh, order = 100.):
  """
  Compute the signed distance field of the polys onto a mesh. The sdf is the
  Euclidean distance between the boundary of the poly and the mesh elements.
  A negative value indicates that the point is inside the poly and a
  positive value indicates the mesh point lies outside the poly.
  Args:
    polys: A dataclass of `ConvexPolys` that describes a set of polys.
    mesh: describes the mesh onto which the sdf is to be computed.
    order: The entries of logsumexp are roughly [-order, order]. This is
      done to ensure that there is no numerical under/overflow.
  Returns: Array of size (num_polys, num_elems) that is the sdf of each poly
    onto the elements of the mesh.
  """

  # b -> poly, s-> side, e -> element
  relative_x = (mesh.elem_centers[:, 0] - polys.center_x[:, np.newaxis])
  relative_y = (mesh.elem_centers[:, 1] - polys.center_y[:, np.newaxis])
  nrml_dot_x = (jnp.einsum('bs, be-> bse', jnp.cos(polys.face_angles), relative_x) + 
        jnp.einsum('bs, be-> bse', jnp.sin(polys.face_angles), relative_y))
  dist_planes = (nrml_dot_x - polys.face_offset[:, :, np.newaxis])

  # implementation issue: The logsumexp has numerical under/over flow issue. To
  # counter this we scale our distances to be roughly by `order` to be 
  # [-order, order]. We multiply the scaling factor outside of LSE and thus
  # get back the correct SDF. This is purely an implementation trick.
  scaling = mesh.lx/order   # we assume lx and ly are roughly in same order
  sdf = scaling*jax.scipy.special.logsumexp(dist_planes/scaling, axis=1)
  return sdf

def impose_poly_symmetry(polys: ConvexPolys):
  """Ensures that each poly is symmetric about the X and Y planes. """
  flipped_offsets = jnp.flip(polys.face_offset, axis=1)
  mean_offset = 0.5*(polys.face_offset + flipped_offsets)
  symm_polys =  ConvexPolys(center_x = polys.center_x,
                           center_y = polys.center_y,
                           angle_offset=polys.angle_offset,
                           face_offset = mean_offset)
  return symm_polys

def erode_polys(polys: ConvexPolys, thickness: float):
  """ Erode polys by given thickness.
  Args:
    polys: The input set of polys which are to be eroded.
    thickness: A positive number that indicates the thickness to which erode the
      polys by. When the thickness exceeds to offset of a face, the offset is
      clipped to zero.
  Returns: A new `ConvexPolys` with the offsets of all the polys and each
    plane of the polys eroded by the specified thickness.
  """
  eroded_polys = ConvexPolys(center_x = polys.center_x,
                           center_y = polys.center_y,
                           angle_offset = polys.angle_offset,
                           face_offset = jnp.clip(polys.face_offset - thickness,
                                                  a_min = 0.))
  return eroded_polys

def dilate_polys(polys: ConvexPolys, thickness: float):
  """Dilate polys by given thickness.
  Args:
    polys: The input set of polys which are to be eroded.
    thickness: A positive number that indicates the thickness to which dilate the
      polys by.
  Returns: A new `ConvexPolys` with the offsets of all the polys and each
    plane of the polys dilated by the specified thickness.
  """
  dilated_polys = ConvexPolys(center_x = polys.center_x,
                           center_y = polys.center_y,
                           angle_offset=polys.angle_offset,
                           face_offset = polys.face_offset + thickness)
  return dilated_polys


class Extremum(Enum):
    MAX = auto()
    MIN = auto()

def smooth_extremum(x: jnp.ndarray,
               order: float = 100.,
               extreme:Extremum = Extremum.MIN):
  """Compute the smooth (approximate) minimum/maximum of a vector.
  Args:
    x: Array of whose entries we wish to compute the minimum.
    order: A float that ensures that the values are scaled appropriately to
      ensure no numerical overflow/underflow. Further, depending upon the
      magnitudes of the entry, experimenting with different values of `order`
      can result in better answers.
    extreme: Whether we wish to compute the minima or the maxima.
  """
  x_nograd = jax.lax.stop_gradient(x)
  scale = np.amax(np.abs(x_nograd))/order
  sgn = -1. if extreme == Extremum.MIN else 1.
  return  scale*sgn*jax.scipy.special.logsumexp(sgn*x/scale)

def compute_min_edge_length_of_polygons(polys: ConvexPolys):
  """Compute the (soft) minimum efge length of the polygons.
    Args:
      polys: A dataclass of `ConvexPolys` that describes a set of polys.
    Returns: A float that is the smallest edge length accounting all the
      polygons and all the sides of each polygon.
  """
  edge_lengths = compute_edge_lengths_of_polygons(polys)
  softmin_edge_length = smooth_extremum(edge_lengths, extreme=Extremum.MIN)
  return softmin_edge_length