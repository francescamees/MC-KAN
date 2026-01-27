from types import SimpleNamespace
import torch

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}


class MCKANPyKANConfig:

    def __init__(
        self,
        n_static_features=1,
        n_dynamic_features=0,
        n_basis=5,
        T=1.0,
        seed=42,
        kan=None,
        training=None,
        dataset_split=None,
        dataloader_type='iterative',
        device='cpu',
        num_epochs=200,
        internal_knots=None,
        n_basis_tunable=False,
        dynamic_bias=False,
        dynamic_agg='mean',
        dynamic_mode='aggregate',
        fusion='add',
        fusion_alpha=0.5,
    ):
        if not isinstance(n_static_features, int) or n_static_features <= 0:
            raise ValueError('n_static_features must be an integer > 0')
        if not isinstance(n_dynamic_features, int) or n_dynamic_features < 0:
            raise ValueError('n_dynamic_features must be an integer >= 0')
        if n_basis < 4:
            raise ValueError('n_basis must be at least 4')
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError("dataloader_type must be one of ['iterative','tensor']")
        if dynamic_agg not in ['mean', 'last']:
            raise ValueError("dynamic_agg must be one of ['mean','last']")
        if dynamic_mode not in ['aggregate', 'history']:
            raise ValueError("dynamic_mode must be one of ['aggregate','history']")
        if fusion not in ['add', 'linear']:
            raise ValueError("fusion must be one of ['add','linear']")
        if fusion == 'linear' and not (0.0 <= fusion_alpha <= 1.0):
            raise ValueError('fusion_alpha must be in [0,1] when fusion=linear')

        if kan is None:
            kan = {
                'widths': [32, 32],
                'grid': 16,
                'degree': 3,
                'kwargs': {},
            }
        if training is None:
            training = {
                'optimizer': 'adam',
                'lr': 1e-3,
                'batch_size': 32,
                'weight_decay': 1e-5,
            }
        if dataset_split is None:
            dataset_split = {'train': 0.8, 'val': 0.1, 'test': 0.1}

        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("training['optimizer'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")

        self.n_static_features = n_static_features
        self.n_dynamic_features = n_dynamic_features
        self.n_basis = n_basis
        self.T = T
        self.seed = seed
        self.kan = SimpleNamespace(**kan)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias
        self.dynamic_agg = dynamic_agg
        self.dynamic_mode = dynamic_mode
        self.fusion = fusion
        self.fusion_alpha = fusion_alpha
