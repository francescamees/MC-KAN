import random
import numpy as np
import torch
from timeview.basis import BSplineBasis
from .config import MCKANPyKANConfig


def _validate_data(X, Zs, ts, ys, T, config):
    assert X.shape[0] == len(ts) == len(ys) == len(Zs)
    for i in range(len(ts)):
        assert ts[i].shape[0] == ys[i].shape[0]
        assert ts[i].max() <= T
        assert ts[i].min() >= 0
    if config.dynamic_mode == 'history':
        first_shape = Zs[0].shape
        for z in Zs:
            if z.shape != first_shape:
                raise ValueError('All dynamic histories must have the same shape when dynamic_mode=history')


def _pad_to_shape(a, shape):
    if a.shape == shape:
        return a
    if len(a.shape) == 1:
        b = np.zeros(shape)
        b[:a.shape[0]] = a
    elif len(a.shape) == 2:
        b = np.zeros(shape)
        b[:a.shape[0], :a.shape[1]] = a
    return b


def _aggregate_dynamic(z, mode='mean'):
    if mode == 'mean':
        return z.mean(axis=0)
    if mode == 'last':
        return z[-1]
    raise ValueError("dynamic_agg must be one of ['mean','last']")


def _prepare_dynamic(zs, config):
    if config.dynamic_mode == 'aggregate':
        return np.stack([_aggregate_dynamic(z, config.dynamic_agg) for z in zs])
    if config.dynamic_mode == 'history':
        flat = []
        first_shape = None
        for z in zs:
            if first_shape is None:
                first_shape = z.shape
            if z.shape != first_shape:
                raise ValueError('All dynamic histories must have the same shape when dynamic_mode=history')
            flat.append(z.reshape(-1))
        return np.stack(flat)
    raise ValueError("dynamic_mode must be one of ['aggregate','history']")


class MCKANDataset(torch.utils.data.Dataset):
    def __init__(self, config, data):
        self.config = config
        T = config.T
        if not isinstance(config, MCKANPyKANConfig):
            raise ValueError('config must be an instance of MCKANPyKANConfig')
        if not isinstance(data, tuple) or len(data) != 4:
            raise ValueError('data must be a tuple (X_static, Zs, ts, ys)')

        X, Zs, ts, ys = data
        _validate_data(X, Zs, ts, ys, T, config)

        self.X = torch.from_numpy(np.array(X)).float()
        Z_prepared = _prepare_dynamic(Zs, config)
        if config.n_dynamic_features != Z_prepared.shape[1]:
            raise ValueError(
                f'n_dynamic_features ({config.n_dynamic_features}) does not match prepared dynamic size ({Z_prepared.shape[1]})'
            )
        self.Z = torch.from_numpy(Z_prepared).float()

        self.ts = ts
        self.ys = ys
        self.D = self.X.shape[0]
        self.Ns = [t.shape[0] for t in ts]
        self.N_max = max(self.Ns)
        self.Phis = self._compute_matrices()

        if self.config.dataloader_type == 'tensor':
            self.Y = torch.stack([torch.from_numpy(_pad_to_shape(
                y, (self.N_max,))).float() for y in self.ys], dim=0)
            self.PHI = torch.stack([torch.from_numpy(_pad_to_shape(
                Phi, (self.N_max, self.config.n_basis))).float() for Phi in self.Phis], dim=0)
            self.NS = torch.tensor(self.Ns)
        else:
            self.Phis = [torch.from_numpy(Phi).float() for Phi in self.Phis]
            self.ys = [torch.from_numpy(y).float() for y in self.ys]

    def _compute_matrices(self):
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        return list(bspline.get_all_matrices(self.ts))

    def __len__(self):
        return self.D

    def __getitem__(self, idx):
        if self.config.dataloader_type == 'iterative':
            return self.X[idx, :], self.Z[idx, :], self.Phis[idx], self.ys[idx]
        return self.X[idx, :], self.Z[idx, :], self.PHI[idx, :, :], self.Y[idx, :], self.NS[idx]

    def get_collate_fn(self):
        def iterative_collate_fn(batch):
            Xs, Zs, Phis, ys = [], [], [], []
            for b in batch:
                Xs.append(b[0])
                Zs.append(b[1])
                Phis.append(b[2])
                ys.append(b[3])
            return torch.stack(Xs, dim=0), torch.stack(Zs, dim=0), Phis, ys

        if self.config.dataloader_type == 'iterative':
            return iterative_collate_fn
        return None


def create_dataloader(config, dataset, indices=None, shuffle=True):
    if not isinstance(config, MCKANPyKANConfig):
        raise ValueError('config must be an instance of MCKANPyKANConfig')
    if not isinstance(dataset, MCKANDataset):
        raise ValueError('dataset must be an instance of MCKANDataset')
    if indices is not None and not isinstance(indices, list):
        raise ValueError('indices must be a list')

    gen = torch.Generator()
    gen.manual_seed(config.seed)

    subset = dataset if indices is None else torch.utils.data.Subset(dataset, indices)
    collate_fn = dataset.get_collate_fn()
    return torch.utils.data.DataLoader(
        subset, batch_size=config.training.batch_size, shuffle=shuffle, generator=gen, collate_fn=collate_fn)


def create_train_val_test_dataloaders(config, dataset):
    if not isinstance(config, MCKANPyKANConfig):
        raise ValueError('config must be an instance of MCKANPyKANConfig')
    if not isinstance(dataset, MCKANDataset):
        raise ValueError('dataset must be an instance of MCKANDataset')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_size = int(config.dataset_split.train * len(dataset))
    val_size = int(config.dataset_split.val * len(dataset))
    test_size = len(dataset) - train_size - val_size

    gen = torch.Generator()
    gen.manual_seed(config.seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=gen)

    collate_fn = dataset.get_collate_fn()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True, generator=gen, worker_init_fn=seed_worker, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False, generator=gen, worker_init_fn=seed_worker, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False, generator=gen, worker_init_fn=seed_worker, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader
