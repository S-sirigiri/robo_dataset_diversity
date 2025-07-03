import h5py
import numpy as np

from maximizer import BlackBoxMaximizer
from maximizer import SubmodularMaximizer
from maximizer import NonMonotoneSubmodularMaximizer

from diversity_metrics import ShannonEntropy
from diversity_metrics import VonNeumannEntropy
from diversity_metrics import VendiScore
from diversity_metrics import LogDeterminant, Determinant

from kernels import KernelMatrix as KernelMatrix

from utils import kern_utils

from rich import print as rprint



class HDF5DatasetReducer:
    def __init__(self, input_path: str):
        """
        Initializes the reducer with the path to the input HDF5 file.
        The output path will automatically be <input_basename>_reduced.hdf5.
        """
        self.input_path = input_path
        self.train_ids = []
        self.data_array = None  # Will hold the (N, T, d) array
        self.index_to_id = {}  # Maps array index -> demo_id
        self.split_data_array = None  # Will hold the (I, N, T, d_i) array (dtype=object)

        self._load_train_ids()
        self._build_padded_array()
        self._build_split_arrays()

    def _load_train_ids(self):
        """Read the train mask from the input file."""
        with h5py.File(self.input_path, 'r') as f:
            raw = f['mask']['train'][()]
            # Convert byte-strings to Python str
            self.train_ids = [s.decode('ascii') for s in raw]

    def _build_padded_array(self):
        """
        Loads each train demo, flattens per-timestep features into a vector,
        and pads all to the maximum trajectory length.
        """
        flat_sequences = []
        max_T = 0

        with h5py.File(self.input_path, 'r') as f:
            for idx, demo_id in enumerate(self.train_ids):
                grp = f['data'][demo_id]

                # Flatten all obs fields
                obs_grp = grp['obs']
                obs_list = [obs_grp[k][()] for k in sorted(obs_grp.keys())]
                obs = np.concatenate(obs_list, axis=1)

                # Flatten all next_obs fields
                nxt_grp = grp['next_obs']
                nxt_list = [nxt_grp[k][()] for k in sorted(nxt_grp.keys())]
                next_obs = np.concatenate(nxt_list, axis=1)

                actions = grp['actions'][()]
                dones   = grp['dones'][()].reshape(-1, 1)
                rewards = grp['rewards'][()].reshape(-1, 1)
                states  = grp['states'][()]

                # Concat into shape (T, d)
                seq = np.concatenate([obs, next_obs, actions, dones, rewards, states], axis=1)
                flat_sequences.append(seq)

                max_T = max(max_T, seq.shape[0])
                self.index_to_id[idx] = demo_id

        N = len(flat_sequences)
        d = flat_sequences[0].shape[1]

        # Pad to shape (N, max_T, d)
        arr = np.zeros((N, max_T, d), dtype=flat_sequences[0].dtype)
        for idx, seq in enumerate(flat_sequences):
            T = seq.shape[0]
            arr[idx, :T, :] = seq
            if T < max_T:
                arr[idx, T:, :] = seq[-1]  # repeat last frame

        self.data_array = arr

    def _build_split_arrays(self):
        """
        Builds self.split_data_array as a single numpy array of shape
        (N, I, T, d_max), where
          N = number of demos,
          I = number of feature‐groups (actions, dones, rewards, states, obs, next_obs),
          T = max trajectory length,
          d_max = max feature‐dimension across those I groups.
        Each feature is padded in time by repeating its last frame, and
        in its feature‐dimension by zeros.
        """
        import h5py, numpy as np

        feature_keys = ['actions', 'dones', 'rewards', 'states', 'obs', 'next_obs']
        feature_keys = ['actions', 'states']
        seqs = {key: [] for key in feature_keys}
        max_T = 0

        # 1) collect each demo’s raw sequence for every feature
        with h5py.File(self.input_path, 'r') as f:
            for demo_id in self.train_ids:
                grp = f['data'][demo_id]

                seqs['actions'].append(grp['actions'][()])
                #seqs['dones'].append(grp['dones'][()].reshape(-1, 1))
                #seqs['rewards'].append(grp['rewards'][()].reshape(-1, 1))
                seqs['states'].append(grp['states'][()])
                """
                obs_grp = grp['obs']
                seqs['obs'].append(np.concatenate(
                    [obs_grp[k][()] for k in sorted(obs_grp.keys())], axis=1
                ))

                nxt_grp = grp['next_obs']
                seqs['next_obs'].append(np.concatenate(
                    [nxt_grp[k][()] for k in sorted(nxt_grp.keys())], axis=1
                ))
                """
                # track maximum trajectory length
                T = seqs['actions'][-1].shape[0]
                if T > max_T:
                    max_T = T

        N = len(self.train_ids)
        I = len(feature_keys)

        # 2) pad each feature‐list into (N, max_T, d_i)
        per_feature = []
        for key in feature_keys:
            sample = seqs[key][0]
            d_i = sample.shape[1]
            arr = np.zeros((N, max_T, d_i), dtype=sample.dtype)

            for idx, seq in enumerate(seqs[key]):
                T = seq.shape[0]
                arr[idx, :T, :] = seq
                if T < max_T:
                    arr[idx, T:, :] = seq[-1]  # repeat last timestep

            per_feature.append(arr)

        # 3) find maximum feature‐dimension
        d_max = max(arr.shape[2] for arr in per_feature)

        # 4) allocate final array and copy each feature into it
        split = np.zeros((N, I, max_T, d_max), dtype=per_feature[0].dtype)
        for i, arr in enumerate(per_feature):
            d_i = arr.shape[2]
            split[:, i, :, :d_i] = arr

        self.split_data_array = split

    def get_demos(self)-> np.ndarray:
        #self.data_array = np.random.permutation(self.data_array)
        return self.split_data_array

    def get_diversity_score(self, metric) -> float:
        return metric(self.split_data_array)

    def reduce(self, top_idxes, metric=None) -> np.ndarray:
        """
        """
        rprint('The dataset ' + "'" + self.input_path + "'" + ' has ' + str(len(self.train_ids)) + ' train demos.')
        if metric is not None:
            rprint(f"Score of original dataset: {metric(self.data_array):.{16}f}")

        reduced_data = self.data_array[list(top_idxes)]

        if metric is not None:
            rprint(f"Score of reduced dataset: {metric(reduced_data):.{16}f}")

        return reduced_data

    def process(self, top_idxes, output_suffix, metric=None):
        """
        Runs reduce(), then writes out a new HDF5 (same name + "_reduced"),
        copying everything except updating only mask/train.
        """
        reduced = self.reduce(top_idxes, metric=metric)
        M = reduced.shape[0]
        #kept_ids = [self.index_to_id[i] for i in range(M)]
        kept_ids = [self.index_to_id[i] for i in top_idxes]
        return self._write_new_file(kept_ids, output_suffix)

    def _write_new_file(self, kept_ids, output_suffix):
        """
        Copy the entire original file to a new one (appending "_reduced"
        to the basename), but overwrite only mask/train.
        """
        base, ext = os.path.splitext(self.input_path)
        out_path = f"{base}_{output_suffix}{ext}"

        with h5py.File(self.input_path, 'r') as src, h5py.File(out_path, 'w') as dst:
            # 1) Copy all top-level groups and datasets (including attrs)
            for name in src:
                if name != 'mask':
                    src.copy(name, dst)

            # 2) Copy root attributes
            for attr_key, attr_val in src.attrs.items():
                dst.attrs[attr_key] = attr_val

            # 3) Rebuild mask/, copying everything except train
            mask_grp = dst.create_group('mask')
            for ds_name in src['mask']:
                if ds_name == 'train':
                    continue
                src['mask'].copy(ds_name, mask_grp)

            # 4) Write updated train mask
            dt = h5py.string_dtype('ascii')
            encoded = [s.encode('ascii') for s in kept_ids]
            mask_grp.create_dataset('train', data=encoded, dtype=dt)

        rprint(f"New dataset written to '{out_path}' with {len(kept_ids)} train demos.")

        return out_path


import argparse
import sys
import os


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
    parser.add_argument('--max-iter', type=int, default=50, help='max_iter for cross-entropy maximizer')
    parser.add_argument('--smoothing', type=float, default=0.7, help='smoothing for cross-entropy maximizer')

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
    reducer = HDF5DatasetReducer(args.input)
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
        if args.kernel_type == 'Random fourier signature features kernel':
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
        rprint(f"The dataset '{args.input}' has {len(data[0])} train demos.")
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
        #if hasattr(km.kernel, 'fit'):
        #    km.kernel.fit(data)
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

    N = len(data[0])
    # Selection
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
            top_idxes = maximizer.greedy_local_search(args.k)
    elif args.maximizer == 'random':
        top_idxes = np.random.choice(N, size=args.k, replace=False).tolist()
    elif args.maximizer == 'arrange':
        top_idxes = np.arange(args.k).tolist()
    else:
        raise ValueError(f"Unknown maximizer: {args.maximizer}")

    # Reduce and write
    if args.get_score_after_maximizing:
        reducer.process(top_idxes, args.output_suffix, metric=metric)
    else:
        reducer.process(top_idxes, args.output_suffix, metric=None)


if __name__ == '__main__':
    main()