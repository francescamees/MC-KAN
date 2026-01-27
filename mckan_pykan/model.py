import torch
from timeview.basis import BSplineBasis


def is_dynamic_bias_enabled(config):
    return getattr(config, 'dynamic_bias', False)


class MCKANPyKAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self._init_encoders()
        if not is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))

    def _init_encoders(self):
        try:
            from kan import KAN
        except Exception as exc:
            raise ImportError(
                'pykan is not installed. Install it and retry, e.g. `pip install pykan`.'
            ) from exc

        out_dim = self.config.n_basis + (1 if is_dynamic_bias_enabled(self.config) else 0)
        static_widths = [self.config.n_static_features] + list(self.config.kan.widths) + [out_dim]
        self.static_encoder = KAN(
            width=static_widths,
            grid=self.config.kan.grid,
            k=self.config.kan.degree,
            seed=self.config.seed,
            **self.config.kan.kwargs,
        )

        if self.config.n_dynamic_features > 0:
            dynamic_widths = [self.config.n_dynamic_features] + list(self.config.kan.widths) + [out_dim]
            self.dynamic_encoder = KAN(
                width=dynamic_widths,
                grid=self.config.kan.grid,
                k=self.config.kan.degree,
                seed=self.config.seed,
                **self.config.kan.kwargs,
            )
        else:
            self.dynamic_encoder = None

    def _combine(self, h_static, h_dyn):
        if h_dyn is None:
            return h_static
        if self.config.fusion == 'add':
            return h_static + h_dyn
        alpha = self.config.fusion_alpha
        return alpha * h_static + (1.0 - alpha) * h_dyn

    def forward(self, X_static, Z_dynamic, Phis):
        h_static = self.static_encoder(X_static)
        h_dyn = None
        if self.dynamic_encoder is not None and Z_dynamic is not None:
            h_dyn = self.dynamic_encoder(Z_dynamic)
        h = self._combine(h_static, h_dyn)

        if is_dynamic_bias_enabled(self.config):
            self.bias = h[:, -1]
            h = h[:, :-1]

        if self.config.dataloader_type == 'iterative':
            if is_dynamic_bias_enabled(self.config):
                return [torch.matmul(Phi, h[d, :]) + self.bias[d] for d, Phi in enumerate(Phis)]
            return [torch.matmul(Phi, h[d, :]) + self.bias for d, Phi in enumerate(Phis)]
        if self.config.dataloader_type == 'tensor':
            if is_dynamic_bias_enabled(self.config):
                return torch.matmul(Phis, torch.unsqueeze(h, -1)).squeeze(-1) + torch.unsqueeze(self.bias, -1)
            return torch.matmul(Phis, torch.unsqueeze(h, -1)).squeeze(-1) + self.bias

    def predict_latent_variables(self, X_static, Z_dynamic=None):
        device = next(self.parameters()).device
        X_static = torch.from_numpy(X_static).float().to(device)
        if Z_dynamic is not None:
            Z_dynamic = torch.from_numpy(Z_dynamic).float().to(device)
        self.eval()
        with torch.no_grad():
            h_static = self.static_encoder(X_static)
            h_dyn = self.dynamic_encoder(Z_dynamic) if self.dynamic_encoder is not None and Z_dynamic is not None else None
            h = self._combine(h_static, h_dyn)
            if is_dynamic_bias_enabled(self.config):
                h = h[:, :-1]
            return h.cpu().numpy()

    def forecast_trajectory(self, x_static, z_dynamic, t):
        device = next(self.parameters()).device
        x_static = torch.unsqueeze(torch.from_numpy(x_static), 0).float().to(device)
        if z_dynamic is not None:
            z_dynamic = torch.unsqueeze(torch.from_numpy(z_dynamic), 0).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)
        self.eval()
        with torch.no_grad():
            h_static = self.static_encoder(x_static)
            h_dyn = self.dynamic_encoder(z_dynamic) if self.dynamic_encoder is not None and z_dynamic is not None else None
            h = self._combine(h_static, h_dyn)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[0, -1]
                h = h[:, :-1]
            return (torch.matmul(Phi, h[0, :]) + self.bias).cpu().numpy()
