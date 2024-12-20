from mpipartition import Partition
from mpipartition import distribute, overload, exchange
import numpy as np

# assuming a cube size of 1.
box_size = 1.0

# partitioning a box with the available MPI ranks
# if no argument is specified, the dimension of the volume is 3
partition = Partition(dimensions=2)

# number of random particles per rank
n_local = 4

# randomly distributed particles in the unit cube
data = {
    "x": np.linspace(partition.origin[0],partition.origin[0]+partition.extent[0],n_local), #np.random.uniform(0, 1, n_local),
    "y": np.linspace(partition.origin[1],partition.origin[1]+partition.extent[1],n_local), #np.random.uniform(0, 1, n_local),
    "id": n_local * partition.rank + np.arange(n_local),
    "rank": np.ones(n_local) * partition.rank
}
#print(partition.rank,data['x'],flush=True)

# assign to rank by position
#data_distributed = distribute(partition, box_size, data, ('x', 'y'))
#print(partition.rank,data_distributed['x'],flush=True)

# make sure we still have all particles
#n_local_distributed = len(data_distributed['x'])
#n_global_distributed = partition.comm.reduce(n_local_distributed)
#if partition.rank == 0:
#    assert n_global_distributed == n_local * partition.nranks

# validate that each particle is in local extent
#bbox = np.array([
#   np.array(partition.origin),
#   np.array(partition.origin) + np.array(partition.extent)
#]).T
#is_valid = np.ones(n_local_distributed, dtype=np.bool_)
#for i, x in enumerate('xy'):
#    is_valid &= data_distributed[x] >= bbox[i, 0]
#    is_valid &= data_distributed[x] < bbox[i, 1]
#assert np.all(is_valid)

"""
overload_length = 0.1
coord_keys = ['x', 'y', 'z']
assert len(coord_keys) == partition.dimensions
print(partition.decomposition)
for i in range(partition.dimensions):
    print(partition.rank,partition.dimensions,partition.decomposition[i])
    assert partition.decomposition[i] > 1  # currently can't overload if only 1 rank
    # we only overload particles in one layer of the domain decomposition
    # so we cannot overload to more than the extent of each partition
    assert overload_length < partition.extent[i] * box_size
"""
data_overloaded = overload(partition, box_size, data, 0.1, ('x', 'y'))
if partition.rank==0: print('\n',data_overloaded['x'])
if partition.rank==0: print('\n',data_overloaded['id'])
if partition.rank==0: print('\n',data_overloaded['rank'])