import torch
import numpy as np

from typing import Set

from kernels import KernelMatrix
from diversity_metrics import VonNeumannEntropy  # Shannon or Von Neumann
#from diversity_metrics import ShannonEntropy as VendiScore  # Shannon or Von Neumann
#from diversity_metrics import LogDeterminant as VendiScore
from diversity_metrics import VendiScore

from maximizer import  SubmodularMaximizer, BlackBoxMaximizer, NonMonotoneSubmodularMaximizer

def test_non_submodular_vendi_top5():
    # generate 10 random trajectories
    trajectories = torch.randn(20, 20, 3).numpy()

    # set up VendiScore (here Shannon)
    km = KernelMatrix()
    vendi = VendiScore(km)#, method='von_neumann')

    # wrap vendi so it handles empty set and does slicing
    """def F(idx_set: Set[int]) -> float:
        print("idx_set", idx_set)
        if len(idx_set) == 0:
            return -float('inf')
        subarr = trajectories[list(idx_set)]
        return vendi(subarr)"""

    """def objective(X, Y=None, diag=False):
        print("X", X)
        if X.shape[0] == 0:
            return 0
        else:
            return vendi(X, Y, diag=diag)"""

    km = KernelMatrix()
    vm = VonNeumannEntropy(km)  # raw von‐Neumann entropy on arrays

    def metric_arr(arr: np.ndarray) -> float:
        if arr.shape[0] == 0:
            return -float('inf')
        return vm(arr)

    #maximizer = BlackBoxMaximizer(metric_arr, trajectories)
    #top5 = maximizer.greedy_local_search(k=5)

    # ground set = indices 0…9
    #all_indices = list(range(len(trajectories)))

    # run the non-submodular random-greedy
    #maximizer = SubmodularMaximizer(vendi, trajectories)
    #top5 = maximizer.lazy_greedy(k=5)

    maximizer = NonMonotoneSubmodularMaximizer(metric_arr, trajectories)
    top5 = maximizer.random_greedy(k=5)

    #maximizer = BlackBoxMaximizer(F, trajectories)
    #top5 = maximizer.greedy_local_search(k=5)

    print("Top 5 values:", trajectories[list(top5)])
    print("Selected indices:", top5)
    #print("Vendi score of selected set:", F(top5))

if __name__ == "__main__":
    test_non_submodular_vendi_top5()