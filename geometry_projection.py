import numpy as np
import jax.numpy as jnp
import jax
import mesher

_mesh = mesher.Mesher

def project_primitive_sdf_to_density(sdf: jnp.ndarray, mesh:_mesh,
                  sharpness: float=10.,
                  order = 10.) -> jnp.ndarray:
  """Projects primitive onto a mesh, given primitive parameters and mesh coords.

  The resulting density field has a value of one when an element intersects
  with a primitive and zero when it lies outside the mesh.

  Args:
    sdf: Array of size (num_objects, num_elems) that is the signed distance
      value for each object as for each element on a mesh. 
    sharpness: The sharpness value controls the slope of the sigmoid function.
      While a larger value makes the transition more sharper, it makes it more
      non-linear.
    order: The sigmoid entries are scaled to roughly [-order, order]. This
      is done to prevent the gradients from dying for large magnitudes of
      the entries.
  Returns:
    density: Array of size (num_objects, num_elems) where the values are in
      range [0, 1] where 0 means the mesh element did not intersect with the
      primitive and 1 means it intersected.
  """
  # the sigmoid function has dying gradients for large values of argument.
  # to avoid this we scale it to the order of `order`. Note that simply scaling
  # doesn't shift the 0 isosurface and hence doesn't mess up or calculations.

  scale = mesh.lx/order  # we assume lx and ly are roughly in same order
  scaled_sdf = sdf/scale
  return jax.nn.sigmoid(-sharpness*scaled_sdf)


def compute_union_density_fields(density: jnp.ndarray, penal = 8.,
                                 x_min = 1e-3) -> jnp.ndarray:
  """Differentiable max function.

  Computes the maximum value of array along specified axis.
  The smooth max scheme is set in the constructor of the class

  Args:
    density: Array of size (num_objects, num_elems) which contain the density of
      each object on the mesh.
    penal: Used in the computation of a penalty based smooth max function,
      the value indicates the p-th norm to take. A larger value while making
      the value closer to the true max value also makes the problem more
      nonlinear.
    x_min: To avoid numerical issues in stiffness matrices used in simulation,
      a small lower bound value > 0 is added to the density.

  Returns: Array of size (num_elems,) which contain the density of the object
  """
  rho = jnp.clip((x_min**penal + 
    (1 - x_min**penal) * jnp.sum(density**penal, axis=0))**(1./penal),
    a_max = 1.)
  return rho

def compute_overlapping_volume(sdf: jnp.ndarray, mesh: _mesh)->float:
  """
  Args:
    sdf: Array of size (num_objects, num_elems) that is the signed distance
      value for each object as for each element on a mesh.
    mesh: A dataclass of `Mesher` which has the `elem_area` and other info.
  Returns: A float indicating the total volume of overlap between the primitives.
  """
  density = project_primitive_sdf_to_density(sdf, mesh)
  return mesh.elem_area*jnp.sum(jax.nn.relu(jnp.sum(density, axis=0) - 1.))