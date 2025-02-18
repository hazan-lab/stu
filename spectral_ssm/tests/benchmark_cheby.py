import math
import time
import torch
import numpy as np
from scipy.integrate import dblquad


# ------------------------------------------------------------------------
# Chebyshev Polynomial Implementations
# ------------------------------------------------------------------------


def poly_mul_x(poly):
    """Multiply polynomial by x."""
    return [0] + poly


def poly_scale(poly, factor):
    """Scale polynomial coefficients."""
    return [coef * factor for coef in poly]


def poly_sub(poly1, poly2):
    """Subtract poly2 from poly1."""
    length = max(len(poly1), len(poly2))
    result = []
    for i in range(length):
        coef1 = poly1[i] if i < len(poly1) else 0
        coef2 = poly2[i] if i < len(poly2) else 0
        result.append(coef1 - coef2)
    return result

def get_windsor_monic_chebyshev_coeffs(n: int) -> torch.Tensor:
    """Get monic Chebyshev T_n(x) via recurrence. Returns coeffs in descending order."""

    def chebyshev_t_int(n: int) -> list[int]:
        if n == 0:
            return [1]
        elif n == 1:
            return [1, 0]

        T0 = [1]  # T_0(x)
        T1 = [1, 0]  # T_1(x)
        for _ in range(2, n + 1):
            T2 = [2 * c for c in T1] + [0]
            d = len(T2) - len(T0)
            padded_T0 = [0] * d + T0
            T2 = [a - b for a, b in zip(T2, padded_T0, strict=True)]
            T0, T1 = T1, T2
        return T2

    coeffs = torch.tensor(chebyshev_t_int(n), dtype=torch.float64)
    if n > 0:
        coeffs /= 2.0 ** (n - 1)
    return coeffs


def get_isabel_monic_chebyshev_coeffs(n: int) -> list[float]:
    """Get monic Chebyshev T_n(x) via binomial sum. Returns coeffs in ascending order."""
    from math import comb

    max_m = n // 2
    coeffs = [0.0] * (n + 1)

    for k in range(max_m + 1):
        partial_sum = 0
        for m in range(k, max_m + 1):
            partial_sum += comb(n, 2 * m) * comb(m, k) * ((-1) ** k)
        exponent = n - 2 * k
        coeffs[exponent] = partial_sum

    if n > 0:
        lead = coeffs[n]
        coeffs = [c / lead for c in coeffs]

    return coeffs


def get_jerry_chebyshev_coeffs(n: int) -> list[float]:
    """Get Chebyshev T_n(x) via manual recurrence. Returns coeffs in ascending order."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    T_nm2 = [1]  # T_0(x)
    T_nm1 = [0, 1]  # T_1(x)
    for _ in range(2, n + 1):
        term = poly_mul_x(T_nm1)
        term = poly_scale(term, 2)
        T_n = poly_sub(term, T_nm2)
        T_nm2, T_nm1 = T_nm1, T_n
    return T_n


def get_jerry_monic_chebyshev_coeffs(n: int) -> list[float]:
    """Get monic Chebyshev T_n(x) via manual recurrence. Returns coeffs in ascending order."""
    coeffs = get_jerry_chebyshev_coeffs(n)
    if n > 0:
        lead = coeffs[-1]
        coeffs = [c / lead for c in coeffs]
    return coeffs


def integral(a, b, beta):
    """Compute integral of z^a * conj(z)^b over truncated unit disk."""
    if a == b:
        return 2 * beta / (a + b + 2)
    return 2 * np.sin((a - b) * beta) / ((a - b) * (a + b + 2))


def vectorized_integral(a, b, beta):
    """Vectorized version of integral computation."""
    diff = a - b
    denom = a + b + 2
    # Compute sin(x)/x = sinc(x) without division by zero
    sinc_term = np.sinc(diff * beta / np.pi)  # numpy's sinc includes the pi factor
    return 2 * beta * np.where(diff == 0, 1 / denom, sinc_term * np.pi / denom)


def Z_numpy(n: int, beta: float, t: int) -> np.ndarray:
    """Original double-loop numpy implementation."""
    matrix_size = t - n
    results = np.zeros((matrix_size, matrix_size), dtype=complex)

    for i in range(matrix_size):
        for j in range(matrix_size):

            def integrand_real(y, x):
                z = x + 1j * y
                val = z**i * np.conjugate(z) ** j
                return val.real

            def integrand_imag(y, x):
                z = x + 1j * y
                val = z**i * np.conjugate(z) ** j
                return val.imag

            def y_bound(x):
                return min(np.sqrt(1 - x**2), beta)

            res_real, _ = dblquad(integrand_real, -1, 1, lambda x: -y_bound(x), y_bound)
            res_imag, _ = dblquad(integrand_imag, -1, 1, lambda x: -y_bound(x), y_bound)
            results[i, j] = res_real + 1j * res_imag

    return results


def Z_numpy_vectorized(n: int, beta: float, t: int) -> np.ndarray:
    """Original vectorized numpy implementation."""
    matrix_size = t - n
    I = np.arange(matrix_size).reshape(matrix_size, 1)
    J = np.arange(matrix_size).reshape(1, matrix_size)
    diff = I - J
    denom = I + J + 2
    # Use numpy's built-in sinc to avoid division by zero
    sinc_term = np.sinc(diff * beta / np.pi)
    return 2 * beta * np.where(diff == 0, 1 / denom, sinc_term * np.pi / denom)


def Z_jerry(n: int, beta: float, t: int) -> np.ndarray:
    """Manual numpy implementation using coefficient expansion."""
    matrix_size = t - n
    poly_coeff = get_jerry_monic_chebyshev_coeffs(n)

    def compute_entry(i, j):
        ans = 0
        for ii in range(n + 1):
            for jj in range(n + 1):
                if poly_coeff[ii] == 0 or poly_coeff[jj] == 0:
                    continue
                ans += poly_coeff[ii] * poly_coeff[jj] * integral(i + ii, j + jj, beta)
        return ans

    results = np.zeros((matrix_size, matrix_size), dtype=complex)
    for i in range(matrix_size):
        for j in range(matrix_size):
            results[i, j] = compute_entry(i, j)
    return results


def Z_jerry_vectorized(n: int, beta: float, t: int) -> np.ndarray:
    """Vectorized manual numpy implementation."""
    matrix_size = t - n
    poly = np.array(get_jerry_monic_chebyshev_coeffs(n))

    I = np.arange(matrix_size).reshape(matrix_size, 1, 1, 1)
    J = np.arange(matrix_size).reshape(1, matrix_size, 1, 1)
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    ii = ii.reshape(1, 1, n + 1, n + 1)
    jj = jj.reshape(1, 1, n + 1, n + 1)

    A = I + ii
    B = J + jj
    int_vals = vectorized_integral(A, B, beta)
    P = poly.reshape(n + 1, 1) * poly.reshape(1, n + 1)

    return np.sum(int_vals * P, axis=(2, 3))


def Z_pytorch(n: int, beta: float, t: int, chunk_size: int = 256) -> torch.Tensor:
    """GPU-accelerated Z matrix computation."""

    def integral_torch(a, b, beta):
        diff = a - b
        denom = a + b + 2
        return torch.where(
            diff == 0, 2 * beta / denom, 2 * torch.sin(diff * beta) / (diff * denom)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix_size = t - n
    poly = torch.tensor(
        get_jerry_monic_chebyshev_coeffs(n), dtype=torch.float32, device=device
    )

    P = torch.outer(poly, poly)[None, None, :, :]

    ii, jj = torch.meshgrid(
        torch.arange(n + 1, device=device, dtype=torch.float32),
        torch.arange(n + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ii = ii.unsqueeze(0).unsqueeze(0)
    jj = jj.unsqueeze(0).unsqueeze(0)

    Z = torch.empty((matrix_size, matrix_size), dtype=torch.complex64, device=device)

    for i_start in range(0, matrix_size, chunk_size):
        i_end = min(i_start + chunk_size, matrix_size)
        i_vals = torch.arange(i_start, i_end, device=device, dtype=torch.float32)
        i_vals = i_vals.view(-1, 1, 1, 1)

        for j_start in range(0, matrix_size, chunk_size):
            j_end = min(j_start + chunk_size, matrix_size)
            j_vals = torch.arange(j_start, j_end, device=device, dtype=torch.float32)
            j_vals = j_vals.view(1, -1, 1, 1)

            A = i_vals + ii
            B = j_vals + jj
            int_vals = integral_torch(A, B, beta)
            chunk_Z = torch.sum(int_vals * P, dim=(2, 3))
            Z[i_start:i_end, j_start:j_end] = chunk_Z.to(torch.complex64)

    return Z


def test_implementations():
    """Compare all implementations."""
    # Part 1: Compare Chebyshev Polynomial Implementations
    print("\nChebyshev Polynomial Implementations")
    print("=" * 40)

    for n in range(5):
        w = get_windsor_monic_chebyshev_coeffs(n).tolist()[::-1]  # ascending
        i = get_isabel_monic_chebyshev_coeffs(n)
        j = get_jerry_monic_chebyshev_coeffs(n)

        print(f"\nDegree n = {n}")
        print("-" * 20)
        print(f"Windsor: {[round(c, 6) for c in w]}")
        print(f"Isabel:  {[round(c, 6) for c in i]}")
        print(f"Jerry:   {[round(c, 6) for c in j]}")

        # Check if implementations match
        max_wi = max(abs(a - b) for a, b in zip(w, i))
        max_wj = max(abs(a - b) for a, b in zip(w, j))
        if max_wi > 1e-10 or max_wj > 1e-10:
            print("Warning: implementations differ!")
            print(f"Max diff (Windsor-Isabel): {max_wi:.2e}")
            print(f"Max diff (Windsor-Jerry):  {max_wj:.2e}")

    # Part 2: Benchmark Z Matrix Implementations
    print("\n\nZ Matrix Implementation Benchmarks")
    print("=" * 40)

    t_values = [16, 32, 64]  # matrix sizes to test
    header = ["Size (t)", "Deg (n)", "Original", "Vec", "Jerry", "Jerry Vec", "PyTorch"]
    print("\n{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*header))
    print("-" * 70)

    for t in t_values:
        n = math.ceil(math.log(t))
        beta = 1 / (64 * n * n)

        # Original numpy
        start = time.perf_counter()
        Z1 = Z_numpy(n, beta, t)
        t1 = time.perf_counter() - start

        # Original vectorized
        start = time.perf_counter()
        Z2 = Z_numpy_vectorized(n, beta, t)
        t2 = time.perf_counter() - start

        # Jerry's numpy
        start = time.perf_counter()
        Z3 = Z_jerry(n, beta, t)
        t3 = time.perf_counter() - start

        # Jerry's vectorized
        start = time.perf_counter()
        Z4 = Z_jerry_vectorized(n, beta, t)
        t4 = time.perf_counter() - start

        # Windsor's PyTorch implementation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = time.perf_counter()
            Z5 = Z_pytorch(n, beta, t).cpu().numpy()
            torch.cuda.synchronize()
            t5 = time.perf_counter() - start
        else:
            t5 = float("inf")
            Z5 = None

        times = [t, n, t1, t2, t3, t4, t5]
        print(
            "{:<10} {:<10} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
                *times
            )
        )

        if Z5 is not None:
            print("\nAccuracy comparison (max difference vs optimized PyTorch version):")
            print("-" * 50)
            print("Original:    {:.2e}".format(np.max(np.abs(Z1 - Z5))))
            print("Vectorized:  {:.2e}".format(np.max(np.abs(Z2 - Z5))))
            print("Jerry:       {:.2e}".format(np.max(np.abs(Z3 - Z5))))
            print("Jerry Vec:   {:.2e}".format(np.max(np.abs(Z4 - Z5))))
            print()


if __name__ == "__main__":
    test_implementations()
