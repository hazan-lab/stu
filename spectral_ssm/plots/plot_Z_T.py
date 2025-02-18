import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import math

from spectral_ssm.models.stu.model import (
    get_monic_chebyshev_coeffs,
    get_complex_hankel_matrix,
    get_complex_spectral_filters,
)

def check_theoretical_bounds(seq_len: int, k: int, rtol: float = 1e-5) -> dict:
    n = int(math.ceil((7 / 6) * math.log2(seq_len)))
    beta = 1.0 / (64 * n * n)

    results = {}

    # 1. Chebyshev polynomial bounds
    p_coeffs = get_monic_chebyshev_coeffs(n)
    max_coeff = torch.max(torch.abs(p_coeffs))
    coeff_bound = 2.0 ** (0.3 * n)
    results["chebyshev"] = {
        "max_coefficient": max_coeff.item(),
        "coefficient_bound": coeff_bound,
        "bound_satisfied": max_coeff <= coeff_bound,
    }

    # 2. Complex Hankel matrix properties
    Z = get_complex_hankel_matrix(seq_len, p_coeffs, beta)
    is_hermitian = torch.allclose(Z, Z.conj().T, rtol=rtol)
    eigenvals = torch.linalg.eigvalsh(Z)
    is_psd = torch.all(eigenvals > -rtol)
    results["hankel"] = {
        "is_hermitian": is_hermitian,
        "is_psd": is_psd,
        "min_eigenval": eigenvals.min().item(),
        "max_eigenval": eigenvals.max().item(),
    }

    # 3. Eigenvalue decay
    j = torch.arange(1, len(eigenvals) + 1)
    c = torch.exp(torch.tensor(math.pi**2 / 4))
    theoretical_bound = (
        2225 * math.asin(beta) * coeff_bound**2 * (1 + math.log(seq_len - n)) *
        c ** (-j / math.log(seq_len - n))
    )
    decay_satisfied = torch.all(eigenvals <= theoretical_bound)
    results["decay"] = {
        "theoretical_bound": theoretical_bound[:k].tolist(),
        "actual_eigenvals": eigenvals[-k:].tolist(),
        "bound_satisfied": decay_satisfied,
    }

    return results

def plot_filter_analysis(seq_len: int, k: int, save_path: Optional[str] = None) -> None:
    """
    Create comprehensive visualization of filter properties:
      1. Top k filters in time domain
      2. Eigenvalue spectrum and decay bounds
      3. Filter correlation matrix
      4. Approximation error curve

    NOTE: We explicitly convert any BFloat16 tensors to float32 for plotting.
    """
    sigma, phi = get_complex_spectral_filters(seq_len, k)

    # Convert tensors to float32 to ensure supported precision for numpy conversion.
    sigma = sigma.float()
    phi = phi.float()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Top K Filters", 
            "Eigenvalue Spectrum", 
            "Filter Correlation Matrix", 
            "Approximation Error"
        ],
    )

    # 1. Plot filters (using the real part)
    x = np.arange(seq_len)
    for i in range(min(k, 8)):  # Plot up to top 8 filters
        y_data = phi[:, i].real.cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_data,
                name=f"Filter {i + 1}",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

    # 2. Plot eigenvalue spectrum
    sigma_abs = sigma.abs().cpu().numpy()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sigma_abs)),
            y=sigma_abs,
            mode="lines",
            name="Eigenvalues",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(type="log", row=1, col=2)

    # 3. Plot correlation matrix
    corr = torch.abs(phi.T @ phi).float().cpu().numpy()
    fig.add_trace(
        go.Heatmap(z=corr, colorscale="Viridis", name="Correlations"),
        row=2,
        col=1,
    )

    # 4. Plot approximation error
    errors = [
        torch.norm(
            phi[:, : i + 1] @ phi[:, : i + 1].conj().T -
            torch.eye(seq_len, dtype=phi.dtype, device=phi.device)
        ).float().item()
        for i in range(k)
    ]
    fig.add_trace(
        go.Scatter(
            x=np.arange(k),
            y=errors,
            mode="lines+markers",
            name="Error",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=True,
        title_text="Complex Spectral Filter Analysis"
    )

    if save_path:
        # Save as PNG if file extension is .png; otherwise fallback to HTML.
        if save_path.lower().endswith(".png"):
            # Requires Kaleido: pip install -U kaleido
            fig.write_image(save_path)
        elif save_path.lower().endswith(".html"):
            fig.write_html(save_path)
        else:
            # Default to PNG if unrecognized extension.
            fig.write_image(save_path + ".png")
    else:
        fig.show()

def analyze_spectral_filters(seq_len: int = 256, k: int = 8) -> None:
    print(f"Analyzing spectral filters (seq_len={seq_len}, k={k})...")

    bounds = check_theoretical_bounds(seq_len, k)

    print("\nTheoretical Guarantees:")
    print("1. Chebyshev Polynomial Bounds:")
    print(f"   Max coefficient: {bounds['chebyshev']['max_coefficient']:.6f}")
    print(f"   Bound satisfied: {bounds['chebyshev']['bound_satisfied']}")

    print("\n2. Complex Hankel Matrix Properties:")
    print(f"   Hermitian: {bounds['hankel']['is_hermitian']}")
    print(f"   Positive semidefinite: {bounds['hankel']['is_psd']}")

    print("\n3. Eigenvalue Decay:")
    print(f"   Bounds satisfied: {bounds['decay']['bound_satisfied']}")
    print(f"   Top eigenvalue: {bounds['decay']['actual_eigenvals'][-1]:.6f}")

    # Save the plot as a PNG
    plot_filter_analysis(seq_len, k, "complex_filter_analysis.png")
    print("\nPlots saved to 'complex_filter_analysis.png'")

if __name__ == "__main__":
    analyze_spectral_filters()
