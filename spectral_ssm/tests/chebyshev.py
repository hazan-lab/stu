import math
import torch
from typing import Tuple

###############################################################################
# Polynomial operations in "descending order":
#   For example, the list [1, 2, 3] represents 1*x^2 + 2*x + 3.
###############################################################################


def poly_mul_x(coeffs: list[int], factor: int) -> list[int]:
    """
    In descending order, multiplying a polynomial by (factor * x)
    means first multiply each coefficient by 'factor', then append a 0 at the end.

    E.g. if coeffs = [1, 2, 3] => x^2 + 2x + 3,
         then poly_mul_x(coeffs, 2) = [2, 4, 6, 0]
         which represents 2*x^3 + 4*x^2 + 6*x + 0.
    """
    scaled = [c * factor for c in coeffs]
    return scaled + [0]


def poly_sub(a: list[int], b: list[int]) -> list[int]:
    """
    Subtract polynomial b from a, where both are in descending order.
    This returns a - b (also in descending order).
    E.g. if a = [4, 0, -1] => 4*x^2 - 1
             b = [1, 2, 3] => x^2 + 2x + 3
         then result = [3, -2, -4] => 3*x^2 - 2*x - 4
    """
    d = len(a) - len(b)
    if d > 0:
        b = [0] * d + b
    elif d < 0:
        a = [0] * (-d) + a
    return [ac - bc for ac, bc in zip(a, b)]


def chebyshev_t_int(n: int) -> list[int]:
    """
    Returns the coefficients of T_n(x) as *exact integers* in descending order.

    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)

    Leading coefficient of T_n(x) is 2^(n-1) for n >= 1, but we capture this
    exactly in integer form.
    """
    if n == 0:
        return [1]  # T0(x) = 1
    elif n == 1:
        return [1, 0]  # T1(x) = x

    # Build up T_n using the recurrence in integer arithmetic
    T0 = [1]  # T0(x)
    T1 = [1, 0]  # T1(x)
    for k in range(2, n + 1):
        T2 = poly_mul_x(T1, 2)  # 2x * T_{k-1}
        T2 = poly_sub(T2, T0)  # subtract T_{k-2}
        T0, T1 = T1, T2
    return T2


def get_monic_chebyshev_coeffs(n: int, debug: bool = True) -> torch.Tensor:
    """
    Construct the n-th Chebyshev polynomial of the first kind T_n(x) exactly
    in integer arithmetic, then convert to float coefficients for
    the *monic* polynomial M_n(x) = T_n(x) / (2^(n-1)).

    Returns the monic coefficients in descending order as a torch.complex128 tensor.
    For instance, for n=2 we get [1, 0, -0.5], representing x^2 - 1/2.
    """
    if debug:
        print(f"\nConstructing T_{n}(x) via integer recurrence...")
    int_coeffs = chebyshev_t_int(n)

    if debug:
        print(f"T_{n} integer coefficients = {int_coeffs}")
        # Leading coefficient of T_n should be 2^(n-1) if n>=1
        if n >= 1:
            print(f"Leading coefficient of T_{n} = {int_coeffs[0]}, expected ~2^(n-1) = {2 ** (n - 1)}")

    # Convert to float (complex128) and scale by 1/(2^(n-1))
    float_coeffs = torch.tensor(int_coeffs, dtype=torch.complex128)
    if n > 0:
        float_coeffs = float_coeffs / (2.0 ** (n - 1))

    # Now the leading term should be exactly 1.0 if n>=1
    if debug:
        print(f"\nScaled to monic, M_{n}(x) = T_{n}(x) / 2^(n-1)")
        print(f"M_{n} coefficients (descending) = {float_coeffs}")
        if n >= 1:
            lead = float_coeffs[0].abs().item()
            print(f"Leading coeff = {lead} (should be 1.0 if n>0)")

    return float_coeffs


def eval_poly(z: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a polynomial at z using Horner's method.
    The coefficients are assumed to be in descending order, i.e.
    coeffs = [a_k, a_{k-1}, ..., a_0] => p(x) = a_k x^k + ... + a_0.
    """
    result = coeffs[0].expand_as(z)
    for c in coeffs[1:]:
        result = result * z + c
    return result


def verify_bounds(n: int, coeffs: torch.Tensor, rtol: float = 1e-5) -> Tuple[bool, str]:
    """
    Verify bounds from Theorem 2 and Lemma 6.
    We:
      1) check coefficient bounds  <= 2^(0.3 n)
      2) check real evaluation |M_n(x)| <= 2^(1-n) for x in [-1,1] (?)
      3) check complex bounds (|z| in some region)
      4) check derivative bound
    """
    print("\nDetailed coefficient analysis:")
    print(f"n = {n}")
    coeff_bound = 2.0 ** (0.3 * n)
    print(f"Theoretical bound (2^(0.3*n)) = {coeff_bound}")
    print("Actual coefficient magnitudes:")
    for i, c in enumerate(coeffs):
        print(f"  |c_{i}| = {abs(c):.6f}")

    # 1. Check coefficient bounds.
    max_coeff = torch.max(torch.abs(coeffs))
    if max_coeff > coeff_bound:
        return False, f"Coefficient bound violated: {max_coeff} > {coeff_bound}"

    # 2. Check real evaluation bound.
    x = torch.linspace(-1, 1, 1000, dtype=torch.float64)
    poly_vals = eval_poly(x, coeffs)
    max_real = torch.max(torch.abs(poly_vals))
    real_bound = 2.0 ** (-(n - 1)) if n > 0 else 1.0
    if max_real > real_bound:
        return False, f"Real evaluation bound violated: {max_real} > {real_bound}"

    # 3 & 4. Check complex bounds.
    beta = 1.0 / (64.0 * n * n) if n > 0 else 1.0
    xs = torch.linspace(-1, 1, 100, dtype=torch.float64)
    ys = torch.linspace(-beta, beta, 100, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    z = torch.complex(X, Y)

    vals = eval_poly(z, coeffs)
    complex_bound = 1.0 / (2.0 ** (n - 2)) if n >= 2 else 1.0
    max_complex = torch.max(torch.abs(vals))
    if max_complex > complex_bound:
        return False, f"Complex evaluation bound violated: {max_complex} > {complex_bound}"

    # Derivative bounds.
    # The derivative M_n'(x) if M_n(x) has descending coefficients [a_n, a_{n-1}, ..., a_0] is:
    #   M_n'(x) = n*a_n*x^(n-1) + (n-1)*a_{n-1}*x^(n-2} + ...
    # so the descending-coeff vector for M_n'(x) is [n*a_n, (n-1)*a_{n-1}, ..., 1*a_1].
    # We'll do that for the entire domain check too.
    if len(coeffs) > 1:
        d_coeffs = torch.tensor([(n - i) * c for i, c in enumerate(coeffs[:-1])], dtype=torch.complex128)
    else:
        d_coeffs = torch.zeros(1, dtype=torch.complex128)
    d_vals = eval_poly(z, d_coeffs)
    max_deriv = torch.max(torch.abs(d_vals))
    deriv_bound = (n * n) / (2.0 ** (n - 1)) if n >= 1 else 0.0

    # Use relative tolerance for comparison
    if max_deriv > deriv_bound * (1 + rtol):
        return False, f"Derivative bound violated: {max_deriv} > {deriv_bound}"

    return True, "All bounds satisfied"


def get_optimal_params(seq_len: int) -> Tuple[int, float]:
    """
    Get 'optimal' parameters from Theorem 2 for your problem domain.
    For example, n ~ (7/6)*log2(seq_len), etc.
    """
    n = int(math.ceil((7 / 6) * math.log2(seq_len)))
    beta = 1.0 / (64.0 * n * n) if n > 0 else 1.0
    return n, beta


# Main test
if __name__ == "__main__":
    seq_len = 2048
    n, beta = get_optimal_params(seq_len)
    print(f"Optimal parameters for seq_len={seq_len}:")
    print(f"n = {n}")
    print(f"β = {beta:.6f}")

    print("\nComputing Chebyshev coefficients (monic) via integer approach...")
    coeffs = get_monic_chebyshev_coeffs(n, debug=True)
    print("\nMonic Chebyshev coefficients (descending):")
    print(coeffs)

    print("\nVerifying theoretical bounds...")
    passed, msg = verify_bounds(n, coeffs)
    print(msg)

    if passed:
        print("\nAll theoretical guarantees from paper verified!")
