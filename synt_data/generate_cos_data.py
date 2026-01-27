import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_cos_dataset(
    n_samples=200,
    n_points=40,
    seed=42,
    T=2.0,
):
    x_min=0.0,
    x_max=2.5,
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=(n_samples,))
    t = np.linspace(0.0, T, n_points)

    rows = []
    for d in range(n_samples):
        # y = (
        #     (1 + 0.3 * x[d]) * np.cos(2 * np.pi * x[d] * t)
        #     + 0.2 * np.cos(4 * np.pi * t)
        #     + 0.1 * np.cos(2 * np.pi * t**2)
        # )
        y = np.cos( (2*np.pi*t) / x[d] )
        for j in range(n_points):
            rows.append(
                {
                    "trajectory_id": d,
                    "x": float(x[d]),
                    "t": float(t[j]),
                    "y": float(y[j]),
                }
            )

    df = pd.DataFrame(rows)
    return df


def save_sample_plot(df, figure_path):
    trajectory_ids = sorted(df["trajectory_id"].unique())
    stride = max(1, len(trajectory_ids) // 5)
    selected_ids = trajectory_ids[::stride][:5]

    plt.figure(figsize=(8, 5))
    for tid in selected_ids:
        traj = df[df["trajectory_id"] == tid]
        x_val = traj["x"].iloc[0]
        plt.plot(traj["t"], traj["y"], label=f"x = {x_val:.2f}")

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Sample trajectories from complex cosine dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = generate_cos_dataset()
    df.to_csv("data/cos_dataset.csv", index=False)
    save_sample_plot(df, "figures/cos_trajectories.png")


if __name__ == "__main__":
    main()
