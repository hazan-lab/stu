import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px

from spectral_ssm.models.stu import get_spectral_filters


def plot_filters(filters: np.ndarray, k_to_show: int = 16) -> go.Figure:
    """
    Plot the top k_to_show filters using Plotly with distinct colors.

    Args:
        filters: Array of shape (seq_len, num_filters)
        k_to_show: Number of filters to display
    """
    fig = go.Figure()
    x = np.arange(1, filters.shape[0] + 1)

    # Combine two palettes for 16+ distinct colors.
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Dark2

    # Normalize each filter to [-1, 1].
    normalized_filters = np.zeros_like(filters[:, :k_to_show])
    for i in range(k_to_show):
        filter_data = filters[:, i]
        max_abs = np.abs(filter_data).max()
        normalized_filters[:, i] = filter_data / max_abs if max_abs != 0 else filter_data

    for i in range(k_to_show):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=normalized_filters[:, i],
                mode="lines",
                name=f"Filter {i + 1}",
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=(f"Position: %{{x}}<br>Value: %{{y:.3f}}<br>Filter: {i + 1}<extra></extra>"),
            )
        )

    fig.update_layout(
        title="Top 16 Normalized Eigenvector Filters",
        xaxis_title="Position",
        yaxis_title="Normalized Value",
        width=1000,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            itemsizing="constant",
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray", range=[-1.1, 1.1]),
    )

    return fig


def main():
    seq_len = 1024
    K = 24
    filters = get_spectral_filters(seq_len=seq_len, K=K, dtype=torch.float64)
    fig = plot_filters(filters, k_to_show=16)
    fig.write_image("top16_filters.png", scale=2)
    print("Saved plot to 'top16_filters.png'.")


if __name__ == "__main__":
    main()
