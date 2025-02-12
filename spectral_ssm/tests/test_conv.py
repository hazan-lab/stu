import torch

from stu.model import conv, flash_conv

try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False

def compare_and_print(name_a, tensor_a, name_b, tensor_b, rtol=1e-3, atol=1e-3):
    """
    Compare two tensors in float32, printing max diff and allclose status.
    """
    tensor_a_float = tensor_a.float()
    tensor_b_float = tensor_b.float()
    same = torch.allclose(tensor_a_float, tensor_b_float, rtol=rtol, atol=atol)
    diff = (tensor_a_float - tensor_b_float).abs().max().item()
    print(f"{name_a} vs {name_b}: allclose={same}, max diff={diff:.6f}")


def analyze_errors(tensor_a, tensor_b):
    """
    Computes mean relative error and 95th percentile of the relative error.
    """
    a_float = tensor_a.float()
    b_float = tensor_b.float()
    abs_err = (a_float - b_float).abs()
    rel_err = abs_err / (a_float.abs() + 1e-6)  # Avoid division by zero

    # Flatten to CPU to avoid strides or large GPU sorting issues
    rel_err_cpu = rel_err.reshape(-1).cpu()

    mean_rel_err = rel_err_cpu.mean().item()
    pct95_rel_err = torch.quantile(rel_err_cpu, 0.95).item()

    print(f"  Mean relative error: {mean_rel_err:.2e}")
    print(f"  95th percentile: {pct95_rel_err:.2e}")


def test_small_numerical():
    """
    Test smaller shapes for the purpose of analyzing numerical imprecision/stats.
    """
    print("===== SMALL NUMERICAL TEST =====")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Example smaller config
    bsz = 1
    seq_len = 512
    d_in = 64
    K = 16

    # Create random inputs
    u = torch.randn(bsz, seq_len, d_in, device=device)
    v = torch.randn(K, d_in, device=device)

    # Choose FFT size
    n = 1
    while n < seq_len:
        n <<= 1

    flash_fft = FlashFFTConv(seqlen=n, dtype=torch.bfloat16).to(device)

    # Naive conv
    u_plus_na_approx, u_minus_na_approx = conv(u, v, n, use_tensordot=True)
    u_plus_na_exact, u_minus_na_exact = conv(u, v, n, use_tensordot=False)

    # Flash conv
    u_plus_flash_approx, u_minus_flash_approx = flash_conv(u, v, flash_fft, use_tensordot=True)
    u_plus_flash_exact, u_minus_flash_exact = flash_conv(u, v, flash_fft, use_tensordot=False)

    print("===== Shape information =====")
    print(f"Naive approx plus:   {u_plus_na_approx.shape}, minus: {u_minus_na_approx.shape}")
    print(f"Flash approx plus:   {u_plus_flash_approx.shape}, minus: {u_minus_flash_approx.shape}")
    print(f"Naive exact plus:    {u_plus_na_exact.shape}, minus: {u_minus_na_exact.shape}")
    print(f"Flash exact plus:    {u_plus_flash_exact.shape}, minus: {u_minus_flash_exact.shape}\n")

    print("===== Approx mode: Naive vs. Flash =====")
    compare_and_print("na_approx_plus", u_plus_na_approx, "flash_approx_plus", u_plus_flash_approx)
    analyze_errors(u_plus_na_approx, u_plus_flash_approx)
    compare_and_print("na_approx_minus", u_minus_na_approx, "flash_approx_minus", u_minus_flash_approx)
    analyze_errors(u_minus_na_approx, u_minus_flash_approx)

    print("\n===== Exact mode: Naive vs. Flash =====")
    compare_and_print("na_exact_plus", u_plus_na_exact, "flash_exact_plus", u_plus_flash_exact)
    analyze_errors(u_plus_na_exact, u_plus_flash_exact)
    compare_and_print("na_exact_minus", u_minus_na_exact, "flash_exact_minus", u_minus_flash_exact)
    analyze_errors(u_minus_na_exact, u_minus_flash_exact)


def test_performance():
    """
    Benchmark larger shapes for performance comparison between naive and flash,
    with or without torch.compile.
    """
    print("\n===== PERFORMANCE TEST =====")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Configs
    bsz = 8
    seq_len = 8192
    d_in = 64
    K = 24

    # Create random inputs
    u = torch.randn(bsz, seq_len, d_in, device=device)
    v = torch.randn(K, d_in, device=device)

    # Choose FFT size
    n = 1
    while n < seq_len:
        n <<= 1

    flash_fft = FlashFFTConv(seqlen=n, dtype=torch.bfloat16).to(device)

    # We'll define small wrappers so we can compile them with the signature `f()`
    def naive_approx_call():
        # Force usage to prevent compiler from skipping
        plus, minus = conv(u, v, n, use_tensordot=True)
        return (plus + minus).sum()

    def naive_exact_call():
        plus, minus = conv(u, v, n, use_tensordot=False)
        return (plus + minus).sum()

    def flash_approx_call():
        plus, minus = flash_conv(u, v, flash_fft, use_tensordot=True)
        return (plus + minus).sum()

    def flash_exact_call():
        plus, minus = flash_conv(u, v, flash_fft, use_tensordot=False)
        return (plus + minus).sum()

    # Compile the naive versions
    naive_approx_compiled = torch.compile(naive_approx_call)
    naive_exact_compiled = torch.compile(naive_exact_call)

    def benchmark(func, label, warmups=3, iters=10):
        import time
        from statistics import mean, stdev

        # Warmup
        for _ in range(warmups):
            _ = func()
            if device == "cuda":
                torch.cuda.synchronize()

        # Timed runs
        durations = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = func()
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.time() - start)

        mean_ = mean(durations)
        stdev_ = stdev(durations) if iters > 1 else 0.0
        print(f"{label} => mean: {mean_:.6f} sec, stdev: {stdev_:.6f} sec")

    print("Running benchmarks on bigger shapes...")

    benchmark(naive_approx_call, "Naive Approx (Eager)")
    benchmark(naive_approx_compiled, "Naive Approx (Compiled)")
    benchmark(naive_exact_call, "Naive Exact (Eager)")
    benchmark(naive_exact_compiled, "Naive Exact (Compiled)")

    benchmark(flash_approx_call, "Flash Approx")
    benchmark(flash_exact_call, "Flash Exact")


def main():
    # 1) Small test for numerical stats
    test_small_numerical()

    # 2) Larger test for performance
    test_performance()

    print("\nDone.")


if __name__ == "__main__":
    main()
