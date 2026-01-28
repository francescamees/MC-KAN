# Multi-Channel Kolmogorov-Arnold Networks (MC-KANs) for Transparent Time Series Forecasting

MC-KAN implementation built on top of PyKAN and the
TIMEVIEW spline basis. It combines static inputs and history inputs, maps them
to spline coefficients with KAN encoders, and then generates trajectories using
TIMEVIEW's B-spline basis.

## What this repository does

- Static and dynamic inputs are encoded by separate KANs.
- The two encoders are fused (+) into spline coefficients.
- The TIMEVIEW B-spline basis maps coefficients into the final trajectory.

