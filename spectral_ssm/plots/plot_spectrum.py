import numpy as np
import matplotlib.pyplot as plt
from models.stu.model import get_hankel

def analyze_and_plot(seq_len=512, k_to_show=32):
    # Generate matrix and compute eigendecomposition
    Z = get_hankel(seq_len, use_hankel_L=False)
    eigenvals, eigenvecs = np.linalg.eigh(Z)

    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvals))[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Scale eigenvectors by eigenvalues^0.25
    scaled_vecs = eigenvecs * np.abs(eigenvals[None, :]) ** 0.25

    # Compute metrics
    max_vals = np.max(np.abs(scaled_vecs), axis=0)
    norms = np.linalg.norm(scaled_vecs, axis=0)
    cumulative_proportion = np.cumsum(norms) / np.sum(norms)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot 1: Max values of scaled eigenvectors
    ax1.semilogy(range(1, k_to_show + 1), max_vals[:k_to_show], "b.-")
    ax1.grid(True)
    ax1.set_title("Maximum Values of Scaled Eigenvectors")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Max Value (log scale)")

    # Add key threshold lines
    key_thresholds = [
        (16, 0.001, "max_val < 0.001\n99.6% of norm"),
        (8, max_vals[7], "94.5% of norm"),
        (4, max_vals[3], "80.5% of norm"),
    ]

    for k, val, label in key_thresholds:
        ax1.axvline(x=k, color="r", linestyle="--", alpha=0.3)
        ax1.axhline(y=val, color="r", linestyle="--", alpha=0.3)
        ax1.annotate(
            label, xy=(k, val), xytext=(k + 1, val * 1.5), arrowprops=dict(facecolor="red", shrink=0.05), alpha=0.7
        )

    # Plot 2: Norms of scaled eigenvectors
    ax2.semilogy(range(1, k_to_show + 1), norms[:k_to_show], "g.-")
    ax2.grid(True)
    ax2.set_title("Norms of Scaled Eigenvectors")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Norm (log scale)")

    # Plot 3: Cumulative proportion
    ax3.plot(range(1, k_to_show + 1), cumulative_proportion[:k_to_show], "r.-")
    ax3.grid(True)
    ax3.set_title("Cumulative Proportion of Total Norm")
    ax3.set_xlabel("K")
    ax3.set_ylabel("Cumulative Proportion")

    # Add key points on cumulative plot
    key_points = [(4, 0.805676, "K=4: 80.5%"), (8, 0.945093, "K=8: 94.5%"), (16, 0.996079, "K=16: 99.6%")]

    for k, val, label in key_points:
        ax3.plot(k, val, "ro")
        ax3.annotate(label, xy=(k, val), xytext=(k + 1, val - 0.05), arrowprops=dict(facecolor="red", shrink=0.05))

    plt.tight_layout()

    # Print analysis
    print("\nKey Insights:")
    print(f"1. First major drop: K=4 (captures {cumulative_proportion[3]:.1%} of norm)")
    print(f"2. Second major drop: K=8 (captures {cumulative_proportion[7]:.1%} of norm)")
    print(f"3. Recommended cutoff: K=16 (captures {cumulative_proportion[15]:.1%} of norm)")
    print("\nMax values at key points:")
    print(f"K=4:  {max_vals[3]:.6f}")
    print(f"K=8:  {max_vals[7]:.6f}")
    print(f"K=16: {max_vals[15]:.6f}")

    plt.show()
    plt.savefig("hankel_spectrum.png")

    return eigenvals, max_vals, norms, cumulative_proportion


# Run the analysis
eigenvals, max_vals, norms, cum_prop = analyze_and_plot()
