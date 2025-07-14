import argparse
import sys

from rich import print as rprint

import numpy as np

from maximizer import BlackBoxMaximizer
from maximizer import SubmodularMaximizer
from maximizer import NonMonotoneSubmodularMaximizer

from diversity_metrics import ShannonEntropy
from diversity_metrics import VonNeumannEntropy
from diversity_metrics import VendiScore
from diversity_metrics import LogDeterminant, Determinant

from kernels import KernelMatrix
from utils import kern_utils

from HDF5_handler import HDF5DatasetReducer


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset reducer CLI')
    parser.add_argument('--input', '-i', required=True, help='Path to input HDF5 file')
    parser.add_argument('--output-suffix', '-o', default='reduced', help='Suffix for output filename')
    parser.add_argument('--get-score', action='store_true', help='Only compute diversity score and exit')
    parser.add_argument('--get-score-after-maximizing', action='store_true', help='Compute diversity score after maximizing')

    # Kernel options
    parser.add_argument('--kernel-type', default='Random fourier signature features kernel',
                        choices=[
                            'Signature kernel torch', 'Signature kernel cupy', 'Signature kernel',
                            'Random fourier signature features kernel',
                            'Random warping series kernel', 'Global alignment kernel'
                        ], help='Type of sequence kernel')
    parser.add_argument('--bandwidth', type=float, default=None, help='Bandwidth for kernel')
    parser.add_argument('--n-levels', type=int, default=5, help='n_levels for Random Fourier Signature Features')
    parser.add_argument('--n-components', type=int, default=100, help='n_components for RFSF or RWS')
    parser.add_argument('--method', default='TRP', choices=['TRP', 'DP', 'TS'], help='Projection method for RFSF')
    parser.add_argument('--max-warp', type=int, default=None, help='max_warp for Random Warping Series')
    parser.add_argument('--stdev', type=float, default=None, help='stdev for Random Warping Series')
    parser.add_argument('--n-features', type=int, default=None, help='n_features for RWS or Global Alignment')
    parser.add_argument('--random-state', type=int, default=None, help='random_state for RWS or random baseline')
    parser.add_argument('--use-gpu', action='store_true', help='Flag to use GPU in RWS')
    parser.add_argument('--max-batch', type=int, default=None, help='max_batch for Signature kernel torch')
    parser.add_argument('--device', default=None, help='Device for Signature kernel torch')
    parser.add_argument('--dyadic-order', type=int, default=1, help='dyadic_order for Signature kernel torch')

    # Diversity metric
    parser.add_argument('--metric', choices=['shannon', 'von_neumann', 'vendi', 'logdet', 'det'],
                        help='Diversity metric to use')
    parser.add_argument('--vendi-method', choices=['shannon', 'von_neumann'], default='shannon',
                        help='Entropy type for VendiScore')

    # Maximizer options
    parser.add_argument('--maximizer', choices=['submodular', 'nonmonotone', 'blackbox', 'random', 'arrange'],
                        help='Selection strategy')
    parser.add_argument('--stochastic-greedy', action='store_true', help='Use stochastic greedy for submodular maximizer')
    parser.add_argument('--cross-entropy', action='store_true', help='Use cross-entropy for blackbox maximizer')
    parser.add_argument('--k', type=int, help='Cardinality constraint (number of demos to select)')
    parser.add_argument('--epsilon', type=float, default=0.01, help='epsilon for stochastic greedy')
    parser.add_argument('--num-samples', type=int, default=1000, help='num_samples for cross-entropy maximizer')
    parser.add_argument('--elite-frac', type=float, default=0.1, help='elite_frac for cross-entropy maximizer')
    parser.add_argument('--max-iter', type=int, default=50, help='max_iter for blackbox maximizers')
    parser.add_argument('--smoothing', type=float, default=0.7, help='smoothing for cross-entropy maximizer')

    # Embedding options
    parser.add_argument('--embedding', choices=['clip', 'dinov2'], default='clip', help='Embedding choice for the images')

    args = parser.parse_args()
    # If only scoring, no further args required
    if args.get_score:
        return args

    # For reduction runs, ensure required args
    if args.maximizer is None:
        parser.error('Reduction requires --maximizer')
    if args.k is None:
        parser.error('Reduction requires --k')
    if args.maximizer != 'random' and args.metric is None:
        parser.error('Reduction requires --metric unless --maximizer random')
    if args.stochastic_greedy and args.maximizer != 'submodular':
        parser.error('--stochastic-greedy requires --maximizer submodular')
    if args.cross_entropy and args.maximizer != 'blackbox':
        parser.error('--cross-entropy requires --maximizer blackbox')
    return args


def main():
    args = parse_args()
    rprint("[red]Loading the dataset\n[/red]")
    reducer = HDF5DatasetReducer(args.input, args.embedding)
    data = reducer.get_demos()

    # Score-only mode
    if args.get_score:
        bandwidth = args.bandwidth or kern_utils.KernelUtilities.compute_bandwidth(data)
        km = KernelMatrix(
            kernel_type=args.kernel_type,
            max_batch=args.max_batch,
            device=args.device,
            dyadic_order=args.dyadic_order,
            bandwidth=bandwidth,
            n_levels=args.n_levels,
            n_components=args.n_components,
            method=args.method,
            max_warp=args.max_warp,
            stdev=args.stdev,
            n_features=args.n_features,
            random_state=args.random_state,
            use_gpu=args.use_gpu
        )
        if hasattr(km.kernel, 'fit'):
            km.kernel.fit(data)
        if args.metric == 'shannon':
            metric = ShannonEntropy(km)
        elif args.metric == 'von_neumann':
            metric = VonNeumannEntropy(km)
        elif args.metric == 'vendi':
            metric = VendiScore(km, method=args.vendi_method)
        elif args.metric == 'logdet':
            metric = LogDeterminant(km)
        else:
            metric = Determinant(km)
        rprint(f"The dataset '{args.input}' has {data.shape[0]} train demos.")
        rprint(f"Score of dataset: {reducer.get_diversity_score(metric):.16f}")
        sys.exit(0)

    # Instantiate kernel + metric for reduction (if needed)
    if args.maximizer != 'random' or True:
        bandwidth = args.bandwidth or kern_utils.KernelUtilities.compute_bandwidth(data)
        km = KernelMatrix(
            kernel_type=args.kernel_type,
            max_batch=args.max_batch,
            device=args.device,
            dyadic_order=args.dyadic_order,
            bandwidth=bandwidth,
            n_levels=args.n_levels,
            n_components=args.n_components,
            method=args.method,
            max_warp=args.max_warp,
            stdev=args.stdev,
            n_features=args.n_features,
            random_state=args.random_state,
            use_gpu=args.use_gpu
        )
        if hasattr(km.kernel, 'fit'):
            km.kernel.fit(data)
        if args.metric == 'shannon':
            metric = ShannonEntropy(km)
        elif args.metric == 'von_neumann':
            metric = VonNeumannEntropy(km)
        elif args.metric == 'vendi':
            metric = VendiScore(km, method=args.vendi_method)
        elif args.metric == 'logdet':
            metric = LogDeterminant(km)
        else:
            metric = Determinant(km)
    else:
        metric = None

    N = data.shape[0]
    # Selection
    rprint(f"[red]\nSelecting maximized subset of size [/red] {args.k}\n")
    if args.maximizer == 'submodular':
        maximizer = SubmodularMaximizer(metric, data)
        if args.stochastic_greedy:
            top_idxes = maximizer.stochastic_greedy(args.k, epsilon=args.epsilon)
        else:
            top_idxes = maximizer.lazy_greedy(args.k)
    elif args.maximizer == 'nonmonotone':
        maximizer = NonMonotoneSubmodularMaximizer(metric, data)
        top_idxes = maximizer.random_greedy(args.k)
    elif args.maximizer == 'blackbox':
        maximizer = BlackBoxMaximizer(metric, data)
        if args.cross_entropy:
            top_idxes = maximizer.cross_entropy_maximizer(
                args.k,
                num_samples=args.num_samples,
                elite_frac=args.elite_frac,
                max_iter=args.max_iter,
                smoothing=args.smoothing
            )
        else:
            top_idxes = maximizer.greedy_local_search(k=args.k, max_iters=args.max_iter)
    elif args.maximizer == 'random':
        top_idxes = np.random.choice(N, size=args.k, replace=False).tolist()
    elif args.maximizer == 'arrange':
        top_idxes = np.arange(args.k).tolist()
    else:
        raise ValueError(f"Unknown maximizer: {args.maximizer}")

    # Reduce and write
    rprint("[red]Saving the new dataset[/red] \n")
    if args.get_score_after_maximizing:
        reducer.process(top_idxes, args.output_suffix, metric=metric)
    else:
        reducer.process(top_idxes, args.output_suffix, metric=None)


if __name__ == '__main__':
    main()