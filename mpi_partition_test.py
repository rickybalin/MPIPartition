import sys
import numpy as np
from time import sleep

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.threads = True # default
from mpi4py import MPI

from mpipartition import Partition

MPI.Init()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

dim = 2
partition = Partition(dimensions=dim, comm=comm, create_neighbor_topo=True)

N = 3
part_origin = partition.origin
part_extent = partition.extent
x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N,dtype=np.double)
y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N,dtype=np.double)
X, Y = np.meshgrid(x, y)
#print(f'[{rank}]: X = {X}, Y = {Y}',flush=True)
print(f'[{rank}]: x = {x}, y = {y}',flush=True)
comm.Barrier()
sleep(0.1)
if rank == 0: print('')

nb_ranks = partition.neighbor_ranks # gets the neighboring ranks assuming periodicity in all directions
print(f'[{rank}]: neighbor ranks = {nb_ranks}',flush=True)
comm.Barrier()
sleep(0.1)
if rank == 0: print('')

#nbs = partition.neighbors
#print(f'[{rank}]: neighbors = {nbs}',flush=True)
#comm.Barrier()
#sleep(0.1)
#if rank == 0: print('')

# I have to build a halo_info array for each rank of shape [num_halo_nodes,3], where
# col 0 = node ID (local reduced) on local rank
# col 1 = node ID (local reduced) on neighbor rank
# col 2 = neighbor rank ID
if rank == 0: print(partition.coordinates, partition.get_neighbor([0, 0]))


