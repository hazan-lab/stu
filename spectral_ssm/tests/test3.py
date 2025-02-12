import torch
import time


def get_toeplitz(x: torch.Tensor, k: int, lower: bool = True) -> torch.Tensor:
    """
    Efficiently construct Toeplitz matrices for each batch and feature.
    For AR-y, we use lower=True to only look at past values.

    Args:
        x: Input tensor of shape (bsz, sl, d)
        k: Number of steps to include in the Toeplitz construction
        lower: If True, construct lower triangular Toeplitz (past values)
               If False, construct upper triangular Toeplitz (future values)

    Returns:
        torch.Tensor: Toeplitz matrices of shape (bsz, sl, k, d)
    """
    bsz, sl, d = x.shape

    # Create row and column indices for constructing the Toeplitz matrix
    row_indices = torch.arange(sl, device=x.device)  # [0, 1, 2, ..., sl-1]
    col_indices = torch.arange(k, device=x.device)  # [0, 1, 2, ..., k-1]

    # Compute relative positions between each output position and its inputs
    # Shape: (sl, k)
    indices = col_indices - row_indices.unsqueeze(1)

    # Create causality mask
    # For lower triangular (past values), we want indices <= 0
    # For upper triangular (future values), we want indices >= 0
    # Shape: (1, sl, k, 1)
    if lower:
        mask = indices.le(0).unsqueeze(0).unsqueeze(-1)
    else:
        mask = indices.ge(0).unsqueeze(0).unsqueeze(-1)

    # Expand input to match desired output shape
    # Shape: (bsz, sl, k, d)
    x_expanded = x.unsqueeze(2).expand(bsz, sl, k, d)

    # Gather values according to the computed indices
    # -indices gives us the correct offsets for gathering past values
    # clamp(min=0) ensures we don't access negative indices
    # Shape: (bsz, sl, k, d)
    shifted = x_expanded.gather(1, (-indices).clamp(min=0).unsqueeze(0).unsqueeze(-1).expand(bsz, sl, k, d))

    # Apply mask to zero out invalid positions
    # Shape: (bsz, sl, k, d)
    result = shifted * mask.to(x.dtype)

    return result


def compute_ar_y(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Matrix-based implementation using Toeplitz structure for the autoregressive equation:
    ŷ_t = sum(M_i^y * y_{t-i}) for i=1 to k_y

    Args:
        M_y (torch.Tensor): Transition weight matrices of shape (d_out, k_y, d_in)
        y_t (torch.Tensor): Values at each time step of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Autoregressive outputs of shape (bsz, sl, d_out)
    """
    d_out, k_y, d_in = M_y.shape
    bsz, sl, _ = y_t.shape

    # Create Toeplitz matrices for the sequence
    y_toeplitz_full = get_toeplitz(y_t, k_y + 1, lower=True)
    y_toeplitz = y_toeplitz_full[:, :, 1:]  # shape: [bsz, sl, k_y, d_in]

    # Reshape y_toeplitz to [bsz, sl, k_y, 1, d_in] for broadcasting
    y_toeplitz_expanded = y_toeplitz.unsqueeze(-2)  # [bsz, sl, k_y, 1, d_in]

    # Reshape M_y to [1, 1, k_y, d_out, d_in] for broadcasting
    M_y_expanded = M_y.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)  # [1, 1, k_y, d_out, d_in]

    # Multiply and sum over k_y and d_in dimensions
    ar_y = (y_toeplitz_expanded @ M_y_expanded.transpose(-1, -2)).sum(dim=2).squeeze(-2)  # [bsz, sl, d_out]

    # Remove the shift since we want to predict current timestep
    output = ar_y

    return output


def compute_ar_y_einsum(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Einsum-based implementation for the autoregressive equation.

    Args:
        M_y (torch.Tensor): Transition weight matrices of shape (d_out, k_y, d_in)
        y_t (torch.Tensor): Values at each time step of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Autoregressive outputs of shape (bsz, sl, d_out)
    """
    # Create Toeplitz matrices for the sequence
    y_toeplitz_full = get_toeplitz(y_t, M_y.shape[1] + 1, lower=True)
    y_toeplitz = y_toeplitz_full[:, :, 1:]  # shape: [bsz, sl, k_y, d_in]

    # Compute einsum
    ar_y = torch.einsum("bskd,dko->bso", y_toeplitz, M_y)

    return ar_y

def get_toeplitz(x: torch.Tensor, k: int, lower: bool = True) -> torch.Tensor:
    """
    Efficiently construct Toeplitz matrices for each batch and feature.
    For AR-y, we use lower=True to only look at past values.

    Args:
        x: Input tensor of shape (bsz, sl, d)
        k: Number of steps to include in the Toeplitz construction
        lower: If True, construct lower triangular Toeplitz (past values)
               If False, construct upper triangular Toeplitz (future values)

    Returns:
        torch.Tensor: Toeplitz matrices of shape (bsz, sl, k, d)
    """
    bsz, sl, d = x.shape

    # Create row and column indices for constructing the Toeplitz matrix
    row_indices = torch.arange(sl, device=x.device)  # [0, 1, 2, ..., sl-1]
    col_indices = torch.arange(k, device=x.device)  # [0, 1, 2, ..., k-1]

    # Compute relative positions between each output position and its inputs
    # Shape: (sl, k)
    indices = col_indices - row_indices.unsqueeze(1)

    # Create causality mask
    # For lower triangular (past values), we want indices <= 0
    # For upper triangular (future values), we want indices >= 0
    # Shape: (1, sl, k, 1)
    if lower:
        mask = indices.le(0).unsqueeze(0).unsqueeze(-1)
    else:
        mask = indices.ge(0).unsqueeze(0).unsqueeze(-1)

    # Expand input to match desired output shape
    # Shape: (bsz, sl, k, d)
    x_expanded = x.unsqueeze(2).expand(bsz, sl, k, d)

    # Gather values according to the computed indices
    # -indices gives us the correct offsets for gathering past values
    # clamp(min=0) ensures we don't access negative indices
    # Shape: (bsz, sl, k, d)
    shifted = x_expanded.gather(1, (-indices).clamp(min=0).unsqueeze(0).unsqueeze(-1).expand(bsz, sl, k, d))

    # Apply mask to zero out invalid positions
    # Shape: (bsz, sl, k, d)
    result = shifted * mask.to(x.dtype)

    return result


def compute_ar_y(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Matrix-based implementation using Toeplitz structure for the autoregressive equation:
    ŷ_t = sum(M_i^y * y_{t-i}) for i=1 to k_y

    Args:
        M_y (torch.Tensor): Transition weight matrices of shape (d_out, k_y, d_in)
        y_t (torch.Tensor): Values at each time step of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Autoregressive outputs of shape (bsz, sl, d_out)
    """
    d_out, k_y, d_in = M_y.shape
    bsz, sl, _ = y_t.shape

    # Create Toeplitz matrices for the sequence
    y_toeplitz_full = get_toeplitz(y_t, k_y + 1, lower=True)
    y_toeplitz = y_toeplitz_full[:, :, 1:]  # shape: [bsz, sl, k_y, d_in]

    # Reshape y_toeplitz to [bsz, sl, k_y, 1, d_in] for broadcasting
    y_toeplitz_expanded = y_toeplitz.unsqueeze(-2)  # [bsz, sl, k_y, 1, d_in]

    # Reshape M_y to [1, 1, k_y, d_out, d_in] for broadcasting
    M_y_expanded = M_y.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)  # [1, 1, k_y, d_out, d_in]

    # Multiply and sum over k_y and d_in dimensions
    ar_y = (y_toeplitz_expanded @ M_y_expanded.transpose(-1, -2)).sum(dim=2).squeeze(-2)  # [bsz, sl, d_out]

    # Remove the shift since we want to predict current timestep
    output = ar_y

    return output


def compute_ar_y_einsum(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Einsum-based implementation for the autoregressive equation.

    Args:
        M_y (torch.Tensor): Transition weight matrices of shape (d_out, k_y, d_in)
        y_t (torch.Tensor): Values at each time step of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Autoregressive outputs of shape (bsz, sl, d_out)
    """
    # Create Toeplitz matrices for the sequence
    y_toeplitz_full = get_toeplitz(y_t, M_y.shape[1] + 1, lower=True)
    y_toeplitz = y_toeplitz_full[:, :, 1:]  # shape: [bsz, sl, k_y, d_in]

    # Compute einsum
    ar_y = torch.einsum("bskd,dko->bso", y_toeplitz, M_y)

    return ar_y


def measure_performance(func, M_y, y_t, warmup_runs=10, timed_runs=100):
    """
    Measures the execution time and peak memory usage of a given function.

    Args:
        func (callable): The function to measure.
        M_y (torch.Tensor): Transition weight matrices.
        y_t (torch.Tensor): Input tensor.
        warmup_runs (int): Number of warm-up runs.
        timed_runs (int): Number of timed runs.

    Returns:
        Tuple[float, int]: Average execution time per run in milliseconds and peak memory in bytes.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Warm-up runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            func(M_y, y_t)
    torch.cuda.synchronize()

    # Start timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(timed_runs):
            func(M_y, y_t)
        end_event.record()

    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)  # Total time for timed_runs
    average_time_ms = elapsed_time_ms / timed_runs

    peak_memory = torch.cuda.max_memory_allocated()

    return average_time_ms, peak_memory


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available. Please run on a CUDA-enabled device.")

    print(f"Using device: {device}")

    # Seed for reproducibility
    torch.manual_seed(42)

    # Define tensor dimensions
    d_out = 768
    k_y = 3
    d_in = 768
    bsz = 8
    sl = 1024

    # Generate random tensors
    M_y = torch.randn(d_out, k_y, d_in, device=device)
    y_t = torch.randn(bsz, sl, d_in, device=device)

    # Define number of warm-up and timed runs
    warmup_runs = 10
    timed_runs = 100

    print(f"\nTensor dimensions:")
    print(f"  d_out: {d_out}")
    print(f"  k_y: {k_y}")
    print(f"  d_in: {d_in}")
    print(f"  batch size (bsz): {bsz}")
    print(f"  sequence length (sl): {sl}")
    print(f"  Warm-up runs: {warmup_runs}")
    print(f"  Timed runs: {timed_runs}")

    # Measure performance for compute_ar_y (gemm version)
    print("\nMeasuring performance for compute_ar_y (GEMM-based implementation)...")
    time_gemm, memory_gemm = measure_performance(compute_ar_y, M_y, y_t, warmup_runs, timed_runs)
    print(f"  Average Time per run: {time_gemm:.4f} ms")
    print(f"  Peak Memory Usage: {memory_gemm / (1024 ** 2):.4f} MB")

    # Measure performance for compute_ar_y_einsum (einsum version)
    print("\nMeasuring performance for compute_ar_y_einsum (Einsum-based implementation)...")
    time_einsum, memory_einsum = measure_performance(compute_ar_y_einsum, M_y, y_t, warmup_runs, timed_runs)
    print(f"  Average Time per run: {time_einsum:.4f} ms")
    print(f"  Peak Memory Usage: {memory_einsum / (1024 ** 2):.4f} MB")

    # Compare results
    print("\n=== Performance Comparison ===")
    faster = "GEMM" if time_gemm < time_einsum else "Einsum"
    more_memory = "GEMM" if memory_gemm > memory_einsum else "Einsum"
    print(
        f"  Faster Implementation: {faster} ({min(time_gemm, time_einsum):.4f} ms vs {max(time_gemm, time_einsum):.4f} ms)"
    )
    print(
        f"  More Memory Intensive: {more_memory} ({max(memory_gemm, memory_einsum) / (1024 ** 2):.4f} MB vs {min(memory_gemm, memory_einsum) / (1024 ** 2):.4f} MB)"
    )


if __name__ == "__main__":
    main()
