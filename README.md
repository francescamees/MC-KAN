# Multi-Channel Kolmogorov-Arnold Networks (MC-KANs) for Transparent Time Series Forecasting

MC-KAN implementation built on top of PyKAN and the
TIMEVIEW spline basis. It combines static inputs and history inputs, maps them
to spline coefficients with KAN encoders, and then generates trajectories using
TIMEVIEW's B-spline basis.

## What this module does

- Static and dynamic inputs are encoded by separate KANs.
- The two encoders are fused (additive or linear fusion) into spline coefficients.
- The TIMEVIEW B-spline basis maps coefficients into the final trajectory.

## Key files

- `mckan_pykan/config.py`
  - Configuration object (`MCKANPyKANConfig`) with model/training/data options.
  - Important options: `dynamic_mode`, `fusion`, `n_basis`, `T`, `internal_knots`,
    `dataloader_type`.
- `mckan_pykan/data.py`
  - `MCKANDataset` prepares static inputs `X`, dynamic inputs `Z`, and spline
    basis matrices `Phis` (from TIMEVIEW's `BSplineBasis`).
  - Supports `dynamic_mode='aggregate'` or `dynamic_mode='history'`.
- `mckan_pykan/model.py`
  - `MCKANPyKAN` defines the static and dynamic KAN encoders.
  - `forward(X_static, Z_dynamic, Phis)` produces trajectories from coefficients
    and spline bases.
  - `forecast_trajectory` is a helper that builds a basis with TIMEVIEW and
    returns a single forecast.
- `mckan_pykan/training.py`
  - Training loop used by notebooks/experiments.
