import os
import h5py
import numpy as np
from scipy.linalg import bandwidth

from maximizer import BlackBoxMaximizer
from maximizer import SubmodularMaximizer
from maximizer import NonMonotoneSubmodularMaximizer

from diversity_metrics import ShannonEntropy
from diversity_metrics import VonNeumannEntropy
from diversity_metrics import VendiScore
from diversity_metrics import LogDeterminant, Determinant

from kernels import KernelMatrix

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
        self.data_array = None    # Will hold the (N, T, d) array
        self.index_to_id = {}     # Maps array index -> demo_id

        self._load_train_ids()
        self._build_padded_array()

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

    def get_demos(self)-> np.ndarray:
        #self.data_array = np.random.permutation(self.data_array)
        return self.data_array

    def get_diversity_score(self, metric) -> float:
        return metric(self.data_array)

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


import sys

def test():
    args = sys.argv[1:]

    reducer = HDF5DatasetReducer(args[0])

    data = reducer.get_demos()
    bandwidth = kern_utils.KernelUtilities.compute_bandwidth(data)

    #km = KernelMatrix(kernel_type="Random fourier signature features kernel", bandwidth=bandwidth)
    #km.kernel.fit(data)
    #km = KernelMatrix(kernel_type='Signature kernel torch', bandwidth=bandwidth, device='cpu', max_batch=250)
    km = KernelMatrix(kernel_type='Signature kernel cupy', bandwidth=bandwidth)
    metric = VendiScore(km)
    #metric = ShannonEntropy(km)

    rprint('The dataset ' + "'" + args[0] + "'" + ' has ' + str(data.shape[0]) + ' train demos.')
    rprint(f"Score of dataset: {reducer.get_diversity_score(metric):.{16}f}")
    exit(0)


if __name__ == "__main__":
    test()
    #reducer = HDF5DatasetReducer('/home/vardhan/Projects/robo_dataset_diversity/robomimic/datasets/lift/ph/low_dim_v15.hdf5')
    reducer = HDF5DatasetReducer('/home/vardhan/Projects/robo_dataset_diversity/robomimic/datasets/lift/mh/low_dim_v15.hdf5')

    data = reducer.get_demos()

    bandwidth = kern_utils.KernelUtilities.compute_bandwidth(data)

    #km = KernelMatrix(kernel_type="Random fourier signature features kernel", bandwidth=bandwidth)
    #km.kernel.fit(data)
    km = KernelMatrix(kernel_type='Signature kernel cupy', bandwidth=bandwidth)
    metric = ShannonEntropy(km)
    #metric = LogDeterminant(km)
    #maximizer = BlackBoxMaximizer(metric, data)
    maximizer = SubmodularMaximizer(metric, data)

    #top_idxes = maximizer.greedy_local_search(k=54)
    top_idxes = maximizer.lazy_greedy(k=54)
    #top_idxes = np.random.choice(270, size=70, replace=False)
    #top_idxes = np.arange(120)

    out_path = reducer.process(top_idxes, '54_examples_vendi_shannon', metric=None)