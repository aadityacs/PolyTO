import jax.numpy as jnp
import jax
import mesher
import numpy as np
import dataclasses
_mesh = mesher.Mesher

@dataclasses.dataclass
class BlobExtents:
  """Hyper-parameters of the size, number and configurations of the polygons.
  Attributes:
    num_blobs: number of polygons that occupy the design domain.
    num_planes_in_a_blob: number of planes that define each polygon.
    min_center_x: smallest x center coordinate the blobs can have.
    min_center_y: smallest y center coordinate the blobs can have.
    max_center_x: largest x center coordinate the blobs can have.
    max_center_y: largest y center coordinate the blobs can have.
    min_face_offset: smallest offset of a plane from the center.
    max_face_offset: largest offset of a plane from the center.
    min_angle_offset: smallest angular offset of a polygon.
    max_angle_offset: largest angular offset of a polygon.
  """
  num_blobs: int
  num_planes_in_a_blob: int
  min_center_x: float
  min_center_y: float
  max_center_x: float
  max_center_y: float
  min_face_offset: float
  max_face_offset: float
  min_angle_offset: float
  max_angle_offset: float

@dataclasses.dataclass
class ConvexBlob:
  """A dataclass to describe convex blobs.
  
  Each blob is described by a set of hyperplanes. The blobs are inspired from:

  Deng, B., etal (2020). CVXNet: Learnable convex decomposition. CVPR, 31–41.
  https://doi.org/10.1109/CVPR42600.2020.00011

  However to ensure that the hyperplanes always form a bounded region the normals
  are fixed and is decided by the number of planes (N) in each blob. We assume the
  normal angle of the first plane is 0, (2pi)/N in the second, and i(2pi)/N of
  the ith plane. Each plane is further offset by a prescribed distance from the
  center of each blob.

  Attributes:
    center_x: Array of size (num_blobs,) with the x-center of the blobs.
    center_y: Array of size (num_blobs,) with the y-center of the blobs.
    angle_offset: Array of size (num_blobs,) of the angle offset of the blobs in
      radians.
    face_offset: Array of size (num_blobs, num_planes_in_a_blob) with the 
      offset of each of the planes of the faces.
  """
  center_x: jnp.ndarray
  center_y: jnp.ndarray
  angle_offset: jnp.ndarray
  face_offset: jnp.ndarray

  # TODO: need to ensure that `num_planes_in_a_blob` is 3 and greater as the
  # simplex in 2D is a triangle. Also need to ensure atleast one polygon is
  # there.

  @property
  def num_free_parameters(self):
    """Number of free parameters for optimization."""
    return self.num_blobs*(self.num_planes_in_a_blob + 3)

  @property
  def face_angles(self):
    """ Array of size (num_blobs, num_planes_in_blob) """
    return (self.angle_offset[:, np.newaxis] + 
             jnp.linspace(0, 2*np.pi, self.num_planes_in_a_blob + 1)[:-1])

  @property
  def num_blobs(self):
    return self.face_offset.shape[0]

  @property
  def num_planes_in_a_blob(self):
    return self.face_offset.shape[1]

  @classmethod
  def from_array(
      cls,
      state_array: jnp.ndarray,
      num_blobs: int,
      num_planes_in_a_blob: int,
  ) -> 'ConvexBlob':

    cx = state_array[0:num_blobs]
    cy = state_array[num_blobs:2*num_blobs]
    ang = state_array[2*num_blobs:3*num_blobs]
    offset = state_array[3*num_blobs:].reshape((num_blobs, num_planes_in_a_blob))

    return ConvexBlob(cx, cy, ang, offset)

  def to_array(self) -> np.ndarray:
    """Converts the `ConvexBlob` into a rank-1 array."""
    return jnp.concatenate([f.reshape((-1)) for f in dataclasses.astuple(self)])

  def to_normalized_array(self, blob_extents: BlobExtents) -> jnp.ndarray:
    """Converts the `ConvexBlob` into a rank-1 array with values normalized."""
    range_cx = (blob_extents.max_center_x - blob_extents.min_center_x)
    cx = (self.center_x - blob_extents.min_center_x)/range_cx

    range_cy = (blob_extents.max_center_y - blob_extents.min_center_y)
    cy = (self.center_y - blob_extents.min_center_y)/range_cy

    range_ang = (blob_extents.max_angle_offset - blob_extents.min_angle_offset)
    ang = (self.angle_offset - blob_extents.min_angle_offset)/range_ang

    range_offset = (blob_extents.max_face_offset - 
               blob_extents.min_face_offset)
    offset = (self.face_offset - blob_extents.min_face_offset)/range_offset

    return jnp.concatenate(( cx.reshape((-1)), cy.reshape((-1)), 
                            ang.reshape((-1)), offset.reshape((-1)) ))

  @classmethod
  def from_normalized_array(cls, state_array: jnp.ndarray,
      blob_extents: BlobExtents)->'ConvexBlob':

    nb = blob_extents.num_blobs
    np = blob_extents.num_planes_in_a_blob 

    range_cx = (blob_extents.max_center_x - blob_extents.min_center_x)
    cx = state_array[0:nb]*range_cx + blob_extents.min_center_x

    range_cy = (blob_extents.max_center_y - blob_extents.min_center_y)
    cy = state_array[nb:2*nb]*range_cy + blob_extents.min_center_y

    range_ang = (blob_extents.max_angle_offset - blob_extents.min_angle_offset)
    ang = state_array[2*nb:3*nb]*range_ang + blob_extents.min_angle_offset

    range_offset = (blob_extents.max_face_offset - blob_extents.min_face_offset)
    
    offset = (state_array[3*nb:]*range_offset + blob_extents.min_face_offset
              ).reshape((nb, np))

    return ConvexBlob(cx, cy, ang, offset)


def init_blob_grid(nx: int, ny: int, blob_extents: BlobExtents):
  """
  NOTE: User ensures that the number of blobs in `blob_extents` is set as
  `nx*ny`.

  Args:
   nx: number of blobs along the x-axis.
   ny: number of blobs along the y-axis.
   blob_extents: dataclass of `BlobExtents` that contain metadata about the
    blobs.
  
  Returns: A set of `nx*ny` equi-spaced and equi-sized `ConvexBlobs`.
  """
  num_blobs = nx*ny
  len_x = np.abs(blob_extents.max_center_x - blob_extents.min_center_x)
  len_y = np.abs(blob_extents.max_center_y - blob_extents.min_center_y)
  del_x = len_x/(4*nx)
  del_y = len_y/(4*ny)
  face_offset = min(del_x, del_y)*np.ones(
      (num_blobs, blob_extents.num_planes_in_a_blob))
  cx = blob_extents.min_center_x + np.linspace(2*del_x, len_x - 2*del_x, nx)
  cy = blob_extents.min_center_y + np.linspace(2*del_y, len_y - 2*del_y, ny)
  [cx_grid,cy_grid] = np.meshgrid(cx, cy)
  mean_ang = 0.5*(blob_extents.max_angle_offset + blob_extents.min_angle_offset)
  ang_offset = mean_ang*np.ones((num_blobs))
  return ConvexBlob(cx_grid.reshape((-1)), cy_grid.reshape((-1)), 
                    ang_offset, face_offset)


def init_random_blobs(blob_extents:BlobExtents, seed: int = 27):
  """Initialize the blobs randomly.

  Args:
    blob_extents: dataclass of `BlobExtents` that contain metadata about the
    blobs.
    seed: Random seed to be used to ensure reproducibility.
  Returns: A set of randomly initialized `ConvexBlobs`.
  """
  key = jax.random.PRNGKey(seed)
  cxkey, cykey, angkey, offkey = jax.random.split(key, 4)
  cx = jax.random.uniform(cxkey, (blob_extents.num_blobs,),
            minval=blob_extents.min_center_x, maxval=blob_extents.max_center_x)
  cy = jax.random.uniform(cykey, (blob_extents.num_blobs,),
            minval=blob_extents.min_center_y, maxval=blob_extents.max_center_y)
  ang = jax.random.uniform(angkey, (blob_extents.num_blobs,),
            minval=blob_extents.min_angle_offset, maxval=blob_extents.max_angle_offset)
  off = jax.random.uniform(offkey, (blob_extents.num_blobs, blob_extents.num_planes_in_a_blob),
            minval=blob_extents.min_face_offset, maxval=blob_extents.max_face_offset)
  mean_offset = 0.5*(blob_extents.min_face_offset + blob_extents.max_face_offset)
  off = mean_offset*jnp.ones((blob_extents.num_blobs, blob_extents.num_planes_in_a_blob))
  return ConvexBlob(cx, cy, ang, off)


def compute_blob_sdf(blobs: ConvexBlob, mesh: _mesh, use_true_max = False,
                     order = 100.):
  """
  Compute the signed distance field of the blobs onto a mesh. The sdf is the
  Euclidean distance between the boundary of the blob and the mesh elements.
  A negative value indicates that the point is inside the primitive and a
  positive value indicates the mesh point lies outside the domain.
  Args:
    blobs: A dataclass of `ConvexBlob` that describes a set of blobs.
    mesh: describes the mesh onto which the sdf is to be computed.
    use_true_max: The logsumexpp function causes rounding of the primitives.
      We may want to use the true max function during say plotting
    order: The entries of logsumexp are roughly [-order, order]. This is
      done to ensure that there is no numerical under/overflow.
  Returns: Array of size (num_blobs, num_elems) that is the sdf of each blob
    onto the elements of the mesh.
  """

  # b -> blob, s-> side, e -> element
  relative_x = (mesh.elem_centers[:, 0] - blobs.center_x[:, np.newaxis])
  relative_y = (mesh.elem_centers[:, 1] - blobs.center_y[:, np.newaxis])
  nrml_dot_x = (jnp.einsum('bs, be-> bse', jnp.cos(blobs.face_angles), relative_x) + 
        jnp.einsum('bs, be-> bse', jnp.sin(blobs.face_angles), relative_y))
  dist_planes = (nrml_dot_x - blobs.face_offset[:, :, np.newaxis])

  # implementation issue: The logsumexp has numerical under/over flow issue. To
  # counter this we scale our distances to be roughly by `order`` to be 
  # [-order, order]. We multiply the scaling factor outside of LSE and thus
  # get back the correct SDF. This is purely an implementation trick.
  scaling = mesh.lx/order   # we assume lx and ly are roughly in same order
  # TODO: Maybe consider using true_max on for forward compute and off for
  # gradient computation. This ensures that the SDF is exact while ensuring
  # gradients are exact.
  if use_true_max:
    sdf = jnp.amax(dist_planes, axis=1)
  else:
    sdf = scaling*jax.scipy.special.logsumexp(dist_planes/scaling, axis=1)
  return sdf

def impose_blob_symmetry(blob: ConvexBlob):
  """Ensures that each blob is symmetric about the X and Y planes. """
  flipped_offsets = jnp.flip(blob.face_offset, axis=1)
  mean_offset = 0.5*(blob.face_offset + flipped_offsets)
  symm_blobs =  ConvexBlob(center_x = blob.center_x,
                           center_y = blob.center_y,
                           angle_offset=blob.angle_offset,
                           face_offset = mean_offset)
  return symm_blobs

def erode_blob(blob: ConvexBlob, thickness: float):
  """ Erode a blob by given thickness.
  Args:
    blob: The input set of blobs which are to be eroded.
    thickness: A positive number that indicates the thickness to which erode the
      blobs by. When the thickness exceeds to offset of a face, the offset is
      clipped to zero.
  Returns: A new `ConvexBlob` with the offsets of all the blobs and each
    plane of the blob eroded by the specified thickness.
  """
  eroded_blob = ConvexBlob(center_x = blob.center_x,
                           center_y = blob.center_y,
                           angle_offset=blob.angle_offset,
                           face_offset = jnp.clip(blob.face_offset - thickness,
                                                  a_min = 0.))
  return eroded_blob

def dilate_blob(blob: ConvexBlob, thickness: float):
  """Dilate a blob by given thickness.
  Args:
    blob: The input set of blobs which are to be eroded.
    thickness: A positive number that indicates the thickness to which dilate the
      blobs by. When the thickness exceeds to offset of a face, the offset is
      clipped to zero
  Returns: A new `ConvexBlob` with the offsets of all the blobs and each
    plane of the blob dilated by the specified thickness.
  """
  dilated_blob = ConvexBlob(center_x = blob.center_x,
                           center_y = blob.center_y,
                           angle_offset=blob.angle_offset,
                           face_offset = blob.face_offset + thickness)
  return dilated_blob