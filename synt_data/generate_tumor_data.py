import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def _tumor_volume_wilkerson(t, age, weight, initial_tumor_volume, dosage):
    g_0 = 2.0
    d_0 = 180.0
    phi_0 = 10.0

    g = g_0 * (age / 20.0) ** 0.5
    d = d_0 * dosage / weight
    phi = 1.0 / (1.0 + np.exp(-dosage * phi_0))

    return initial_tumor_volume * (phi * np.exp(-d * t) + (1.0 - phi) * np.exp(g * t))


def _tumor_volume_geng(t, age, weight, initial_tumor_volume, start_time, dosage):
    rho_0 = 2.0
    k_0 = 1.0
    k_1 = 0.01
    beta_0 = 50.0
    gamma_0 = 5.0
    v_min = 0.001

    rho = rho_0 * (age / 20.0) ** 0.5
    k = k_0 + k_1 * weight
    beta = beta_0 * (age / 20.0) ** (-0.2)

    def chemo(ti):
        return np.where(ti < start_time, 0.0, dosage * np.exp(-gamma_0 * (ti - start_time)))

    def dVdt(v, ti):
        return rho * (v - v_min) * v * np.log(k / v) - beta * v * chemo(ti)

    return odeint(dVdt, initial_tumor_volume, t)[:, 0]


def generate_tumor_dataset(
    n_samples=200,
    n_time_steps=40,
    time_horizon=2.0,
    noise_std=0.0,
    seed=0,
    equation="wilkerson",
):
    feature_ranges = {
        "age": (20.0, 80.0),
        "weight": (40.0, 100.0),
        "initial_tumor_volume": (0.1, 0.5),
        "start_time": (0.0, 1.0),
        "dosage": (0.0, 1.0),
    }

    rng = np.random.default_rng(seed)

    age = rng.uniform(*feature_ranges["age"], size=n_samples)
    weight = rng.uniform(*feature_ranges["weight"], size=n_samples)
    tumor_volume = rng.uniform(*feature_ranges["initial_tumor_volume"], size=n_samples)
    start_time = rng.uniform(*feature_ranges["start_time"], size=n_samples)
    dosage = rng.uniform(*feature_ranges["dosage"], size=n_samples)

    if equation == "wilkerson":
        static_cols = ["age", "weight", "initial_tumor_volume", "dosage"]
        X = np.stack((age, weight, tumor_volume, dosage), axis=1)
    elif equation == "geng":
        static_cols = ["age", "weight", "initial_tumor_volume", "start_time", "dosage"]
        X = np.stack((age, weight, tumor_volume, start_time, dosage), axis=1)
    else:
        raise ValueError("equation must be 'wilkerson' or 'geng'")

    ts = np.linspace(0.0, time_horizon, n_time_steps)

    rows = []
    for i in range(n_samples):
        if equation == "wilkerson":
            age_i, weight_i, vol_i, dose_i = X[i, :]
            y = _tumor_volume_wilkerson(ts, age_i, weight_i, vol_i, dose_i)
        else:
            age_i, weight_i, vol_i, start_i, dose_i = X[i, :]
            y = _tumor_volume_geng(ts, age_i, weight_i, vol_i, start_i, dose_i)

        if noise_std > 0.0:
            y = y + rng.normal(0.0, noise_std, size=n_time_steps)

        for j in range(n_time_steps):
            row = {
                "trajectory_id": i,
                "t": float(ts[j]),
                "y": float(y[j]),
            }
            for col, val in zip(static_cols, X[i, :]):
                row[col] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = ["trajectory_id"] + static_cols + ["t", "y"]
    return df[ordered_cols]


def prepare_static_history(df, history_end=1.0):
    static_cols = [c for c in df.columns if c not in ("trajectory_id", "t", "y")]
    trajectory_ids = sorted(df["trajectory_id"].unique())

    X_static = (
        df.drop_duplicates("trajectory_id")
        .sort_values("trajectory_id")[static_cols]
        .to_numpy()
    )

    Zs = []
    ts = []
    ys_list = []
    for tid in trajectory_ids:
        traj = df[df["trajectory_id"] == tid].sort_values("t")
        t_vals = traj["t"].to_numpy()
        y_vals = traj["y"].to_numpy()

        hist_mask = t_vals <= history_end
        pred_mask = t_vals > history_end

        Zs.append(y_vals[hist_mask].reshape(-1, 1))
        ts.append(t_vals[pred_mask].copy())
        ys_list.append(y_vals[pred_mask].copy())

    return X_static, Zs, ts, ys_list


def save_sample_plot(df, figure_path):
    trajectory_ids = sorted(df["trajectory_id"].unique())
    stride = max(1, len(trajectory_ids) // 5)
    selected_ids = trajectory_ids[::stride][:5]

    plt.figure(figsize=(8, 5))
    for tid in selected_ids:
        traj = df[df["trajectory_id"] == tid]
        plt.plot(traj["t"], traj["y"], label=f"id = {tid}")

    plt.xlabel("t")
    plt.ylabel("tumor volume")
    plt.title("Sample trajectories from synthetic tumor dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = generate_tumor_dataset()
    df.to_csv("data/tumor_dataset.csv", index=False)
    save_sample_plot(df, "figures/tumor_trajectories.png")


if __name__ == "__main__":
    main()
