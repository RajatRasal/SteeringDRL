from attribute_disentanglement.datasets import cars3d
from attribute_disentanglement.datasets import mpi3d
from attribute_disentanglement.datasets import shapes3d


def get_named_ground_truth_data(name):
  """Returns ground truth data set based on name.

  Args:
    name: String with the name of the dataset.

  Raises:
    ValueError: if an invalid data set name is provided.
  """
  if name == "cars3d":
    return cars3d.Cars3D()
  elif name == "mpi3d":
    return mpi3d.MPI3D(mode="mpi3d_toy")
  elif name == "mpi3d_realistic":
    return mpi3d.MPI3D(mode="mpi3d_realistic")
  elif name == "mpi3d_real":
    return mpi3d.MPI3D(mode="mpi3d_real")
  elif name == "shapes3d":
    return shapes3d.Shapes3D()
  else:
    raise ValueError("Invalid data set name.")
