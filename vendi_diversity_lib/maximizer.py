import math
import heapq
from typing import Callable, Iterable, TypeVar, Set, List

import numpy as np
rng = np.random.default_rng()

from rich import print as rprint

# Generic type variable for elements in the ground set
T = TypeVar('T')

class _Maximizer:
    """
    Base maximizer that wraps an objective metric into a set-function form
    and initializes the ground set over trajectory indices.
    """

    def __init__(self, objective_metric: Callable[[np.ndarray], float], universe_set: np.ndarray):
        # Ground set: list of indices corresponding to trajectories
        self.X = list(range(len(universe_set)))

        # Define a set-based objective: returns 0.0 for empty sets, else slices the universe
        def F(idx_set: Set[int]) -> float:
            if not idx_set:
                return -1000
            subarr = universe_set[list(idx_set)]
            return objective_metric(subarr)

        self.F = F

class SubmodularMaximizer(_Maximizer):
    """
    Implements greedy algorithms for monotone submodular functions.
    """

    def __init__(self, F: Callable[[Set[T]], float], X: Iterable[T]):
        super().__init__(F, X)

    def lazy_greedy(self, k: int) -> Set[T]:
        """
        Run the lazy (accelerated) greedy algorithm.
        Picks up to k elements with largest marginal gains.
        """
        S: Set[T] = set()                  # Selected set
        S_val = self.F(S)                 # Current objective value
        heap: List[tuple] = []            # Max-heap storing (-gain, element)
        empty_val = self.F(set())         # Value at empty set

        # Initialize heap with singleton marginal gains
        for e in self.X:
            gain = self.F({e}) - empty_val
            heapq.heappush(heap, (-gain, e))

        # Greedily select up to k elements
        for i in range(k):
            rprint('Iteration: {}'.format(i))
            rprint('Objective value: {} \n'.format(S_val))
            while heap:
                neg_gain, e = heapq.heappop(heap)
                # Recompute true marginal gain for current S
                current_gain = self.F(S | {e}) - S_val
                # If cached gain matches recomputed gain (within tolerance), select
                if abs(-neg_gain - current_gain) < 1e-9:
                    S.add(e)
                    S_val += current_gain
                    break
                # Otherwise, push updated gain back into heap
                heapq.heappush(heap, (-current_gain, e))
            else:
                # No more profitable elements
                break

        return S

    def stochastic_greedy(self, k: int, epsilon: float = 0.01) -> Set[T]:
        """
        Run the stochastic greedy algorithm.
        Randomly samples a subset of candidates of size r each iteration.
        """
        S: Set[T] = set()
        S_val = self.F(S)
        n = len(self.X)
        # Sample size r for desired accuracy-speed trade-off
        r = math.ceil((n / k) * math.log(1 / epsilon))

        for i in range(k):
            rprint('Iteration: {}'.format(i))
            rprint('Objective value: {} \n'.format(S_val))
            # Determine remaining elements not yet selected
            remaining = [e for e in self.X if e not in S]
            if not remaining:
                break

            # Sample without replacement from remaining
            sample = list(rng.choice(remaining,
                                     size=min(len(remaining), r),
                                     replace=False))
            best_e, best_gain = None, -float('inf')

            # Evaluate marginal gains on the sample
            for e in sample:
                gain = self.F(S | {e}) - S_val
                if gain > best_gain:
                    best_gain, best_e = gain, e

            # Add the element with positive gain if found
            #if best_gain > 0 and best_e is not None:
            #    S.add(best_e)
            #    S_val += best_gain
            if best_e is None:
                break
            S.add(best_e)
            S_val += best_gain

        return S

class NonMonotoneSubmodularMaximizer(_Maximizer):
    """
    Implements random-greedy for possibly non-monotone functions.
    """

    def __init__(self, F: Callable[[Set[T]], float], X: Iterable[T]):
        super().__init__(F, X)

    def random_greedy(self, k: int) -> Set[T]:
        """
        Discrete Random-Greedy algorithm:
        1) Compute marginal gains Δ(e|S) for all e ∉ S.
        2) Sort and take top-k candidates by gain.
        3) Pick one uniformly at random from the top-k.
        Repeats for k iterations, offering a 1/e approximation.
        """
        S: Set[T] = set()           # Selected set
        S_val = self.F(S)          # Current objective value

        for i in range(k):
            rprint('Iteration: {}'.format(i))
            rprint('Objective value: {} \n'.format(S_val))
            # Compute marginal gains for all remaining candidates
            remaining = [e for e in self.X if e not in S]
            margins = [(e, self.F(S | {e}) - S_val) for e in remaining]

            if not margins:
                break

            # Sort candidates by gain descending and select top-k
            margins.sort(key=lambda x: x[1], reverse=True)
            if len(margins) < k:
                margins += [(None, 0.0)] * (k - len(margins))
            top_k = margins[:k]

            # Sample one candidate uniformly among the top-k
            idx = rng.integers(0, len(top_k))
            e_star, gain = top_k[idx]
            if e_star is None or (gain <= 0 and _ > 1):
                break

            # Add chosen element to the set
            S.add(e_star)
            S_val += gain

        return S


class BlackBoxMaximizer(_Maximizer):
    """
    Black‐box maximizer under a cardinality k constraint.
    """
    def greedy_local_search(self, k: int, max_iters: int = 0) -> Set[T]:
        """
        1) Greedy initialization: pick up to k elements with largest marginal gains.
        2) Local search: while possible, swap one selected element with one unselected
           element if it strictly increases the objective, up to max_iters rounds.
        """
        # --- Greedy initialization ---
        S: Set[T] = set()
        S_val = self.F(S)
        for i in range(k): #for i in range(min(k, len(self.X))):
            rprint('Iteration: {}'.format(i))
            rprint('Objective value: {} \n'.format(S_val))
            best_e, best_gain = None, -float('inf')
            for e in self.X:
                if e in S:
                    continue
                gain = self.F(S | {e}) - S_val
                if gain > best_gain:
                    best_gain, best_e = gain, e
            #if best_e is None or best_gain <= 0:
            #    break
            if best_e is None:
                break
            S.add(best_e)
            S_val += best_gain

        # --- 1‐Swap Local Search ---
        iters = 0
        improved = True
        while improved and iters < max_iters:
            improved = False
            for e in list(S):
                for f in self.X:
                    if f in S:
                        continue
                    new_S = (S - {e}) | {f}
                    new_val = self.F(new_S)
                    if new_val > S_val:
                        S, S_val = new_S, new_val
                        improved = True
                        break
                if improved:
                    break
            iters += 1
        return S


    def cross_entropy_maximizer(
        self,
        k: int,
        num_samples: int = 1000,
        elite_frac: float = 0.1,
        max_iter: int = 50,
        smoothing: float = 0.7
    ) -> Set[T]:
        """
        Cross-Entropy method for maximizing a black-box set function under a
        cardinality constraint of size k.

        Args:
            k: number of elements to include in the solution set.
            num_samples: number of candidate sets sampled per iteration.
            elite_frac: fraction of top samples used to update the sampling distribution.
            max_iter: maximum number of CE iterations.
            smoothing: probability smoothing factor (between 0 and 1).

        Returns:
            A set of k elements approximating the maximum.
        """
        n = len(self.X)
        # Initialize inclusion probabilities uniformly
        p = np.ones(n) * (k / n)

        best_S: Set[T] = set()
        best_val = -float('inf')

        for _ in range(max_iter):
            samples = []  # list of index arrays
            vals = []     # list of objective values

            # Sampling phase
            for _ in range(num_samples):
                # Draw k distinct indices according to probabilities p
                probs = p / p.sum()
                idxs = np.random.choice(n, size=k, replace=False, p=probs)
                S = {self.X[i] for i in idxs}
                val = self.F(S)
                samples.append(idxs)
                vals.append(val)

                # Track global best
                if val > best_val:
                    best_val = val
                    best_S = S

            # Select elite samples
            elite_count = max(1, int(elite_frac * num_samples))
            elite_idxs = np.argsort(vals)[-elite_count:]

            # Update probabilities based on elite samples
            new_p = np.zeros(n)
            for ei in elite_idxs:
                for i in samples[ei]:
                    new_p[i] += 1
            # Normalize to match expected cardinality
            new_p = new_p / elite_count

            # Smooth update
            p = smoothing * p + (1 - smoothing) * new_p

        return best_S


class DeterminantalPointProcessesSampling:
    """
    Sample a diverse subset of fixed size k via k-DPP.

    API:
      sampler = DeterminantalPointProcessesSampling(kernel, data)
      S = sampler.sample(k)
      # S is a set of indices of length k
    """

    def __init__(self, kernel: callable, data: np.ndarray):
        """
        Parameters
        ----------
        kernel : callable
            A kernel function with signature kernel(X, Y=None, diag=False),
            returning an (N,N) array when called as kernel(X, X).
        data : np.ndarray, shape (N, ...)
            The items over which to sample.
        """
        self.data = data
        # Compute the L-ensemble kernel matrix
        self.L = kernel(data, data)
        self.L = self.L / 1e-20
        # Eigendecomposition of L (symmetric)
        self.pi, self.V = np.linalg.eigh(self.L)

    def sample(self, k: int) -> set:
        """
        Draw a random subset of size k using the k-DPP algorithm.

        Returns
        -------
        S : set of int
            Indices of the selected subset of size k.
        """
        N = len(self.pi)
        # STEP 1: Compute elementary symmetric polynomials e[n, l]
        e = np.zeros((N+1, k+1))
        e[:, 0] = 1.0
        for l in range(1, k+1):
            for n in range(1, N+1):
                e[n, l] = e[n-1, l] + self.pi[n-1] * e[n-1, l-1]

        # STEP 2: Sample eigenvector indices J (size k)
        J = []
        rem = k
        for n in range(N, 0, -1):
            if rem == 0:
                break
            # Force-pick if the remaining slots equal remaining eigenvectors
            if rem == n:
                J.extend(range(n-1, -1, -1)[:rem])
                break
            prob = (self.pi[n-1] * e[n-1, rem-1]) / e[n, rem]
            if np.random.rand() < prob:
                J.append(n-1)
                rem -= 1

        # Ensure exactly k selected
        if len(J) != k:
            raise ValueError(f"Expected to select {k} eigenvectors, but got {len(J)}")

        # STEP 3: Sample actual items via the selected eigenvectors
        V_sub = self.V[:, J].copy()  # shape (N, k)
        S = []
        tol = 1e-12
        for _ in range(k):
            # 3a) Compute marginal probabilities
            probs = np.sum(V_sub**2, axis=1)
            probs /= probs.sum()
            i = np.random.choice(N, p=probs)
            S.append(i)
            # 3b) Zero out chosen row
            V_sub[i, :] = 0.0
            # 3c) Remove nearly zero columns
            col_norms = np.linalg.norm(V_sub, axis=0)
            keep = col_norms > tol
            V_sub = V_sub[:, keep]
            if V_sub.size == 0:
                break
            # 3d) Orthonormalize columns via QR
            V_sub, _ = np.linalg.qr(V_sub, mode='reduced')

        if len(S) != k:
            raise RuntimeError(f"k-DPP sampling returned {len(S)} items, expected {k}")

        return set(S)