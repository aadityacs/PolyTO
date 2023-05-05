import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

def plot_bar_primitives(fig, primitives, primitive_params) -> None:
  figu = plt.figure(fig)    
  plt.clf()
  # set the color of the bars
  bar_color = np.zeros((primitive_params.num_bars, 3))
  bar_color[:,0] = 1. # red

  start_pts = jnp.vstack((primitives.start_x, primitives.start_y))
  end_pts = jnp.vstack((primitives.end_x, primitives.end_y))

  verts = np.stack((start_pts, end_pts), axis = 1).T
  lc = matplotlib.collections.LineCollection(verts,\
      linewidths = primitives.radius,\
      color = bar_color)
  ax = plt.gca()   
  ax.cla()
  ax.add_collection(lc)
  
  xy_min = primitive_params.xy_min
  xy_max = xy_min + primitive_params.xy_range
  plt.xlim( (xy_min[0], xy_max[0]) )
  plt.ylim( (xy_min[1], xy_max[1]) )
  plt.gca().set_aspect('equal', adjustable='box')

  plt.pause(0.0001)
  plt.draw()

def plot_density(density:jnp.ndarray, fig:int, titleStr:str):
  plt.ion()
  figu = plt.figure(fig)
  ax = plt.gca()

  ax.cla()
  fv = plt.imshow(density.T, cmap = 'rainbow', origin = 'lower')
  plt.gca().set_aspect('equal', adjustable='box')
  plt.colorbar(fv)
  plt.title( titleStr )    
  plt.pause(0.0001)
  plt.draw()


def plot_history(history):
  for key in history:
    plt.figure()
    plt.plot(history[key].T.reshape((-1)))
    plt.xlabel('iter')
    plt.ylabel(key)