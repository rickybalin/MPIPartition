from mpipartition import Partition
from mpipartition import distribute, overload, exchange
import numpy as np

# assuming a cube size of 1.
box_size = 1.0

# partitioning a box with the available MPI ranks
# if no argument is specified, the dimension of the volume is 3
partition = Partition(dimensions=2, create_neighbor_topo=True)
neighbors = partition.neighbor_ranks # list -- gets the neighboring ranks assuming periodicity in all directions
n_neighbors = len(neighbors)

# number of random particles per rank
n_local = 4

# Create grid
x = np.linspace(partition.origin[0],partition.origin[0]+partition.extent[0],num=n_local,dtype=np.double)
y = np.linspace(partition.origin[1],partition.origin[1]+partition.extent[1],num=n_local,dtype=np.double)
X, Y = np.meshgrid(x, y)
data = {
    "x": X.flatten(), #np.random.uniform(0, 1, n_local),
    "y": Y.flatten(), #np.random.uniform(0, 1, n_local),
    "id": n_local**2 * partition.rank + np.arange(n_local**2),
    "rank": np.ones(n_local**2, dtype=np.int64) * partition.rank
}

# Overload the partitions
data_overloaded = overload(partition, box_size, data, 0.1, ('x', 'y'))

neighbors2 = np.unique(data_overloaded['rank'])
neighbors2 = np.delete(neighbors2,np.argwhere(neighbors2==partition.rank).item())
assert np.allclose(np.array(neighbors),neighbors2)

# Create halo_info of shape [num_halo,3], where
# col 0 = node ID (local reduced) on local rank
# col 1 = node ID (local reduced) on neighbor rank
# col 2 = neighbor rank ID
idx = np.argwhere(data_overloaded['rank'] != partition.rank)
n_halo = len(idx)
assert data_overloaded['rank'].size == n_local**2+n_halo
halo_info = np.hstack((
    idx,
    data_overloaded['id'][idx]-data_overloaded['rank'][idx]*n_local**2,
    data_overloaded['rank'][idx]
))
print(partition.rank,' Number of halo nodes ',n_halo,flush=True)
print(partition.rank,' halo_info ',halo_info,flush=True)

# Create node degree of shape (n_local+n_halo) with number of times a node is repeated across all parts
# could create a mask on each rank with 0 if global ID is not present and 1 if it is present, then do an allreduce sum on the mask and count

# Create edge weight of shape (n_edges) with number of times an edge is repeated across all parts