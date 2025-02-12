import torch
import time
from typing import Tuple, Callable

def eval_poly(z: complex, coeffs: torch.Tensor) -> complex:
    """Original polynomial evaluation"""
    n = len(coeffs) - 1
    result = 0j
    for k in range(n + 1):
        c_k = coeffs[k]
        power = n - k  # decreasing powers
        result += c_k * (z ** power)
    return result

def hankel_brute(
    p_coeffs: torch.Tensor,
    T: int,
    beta: float,
    Nxy: int = 601
) -> torch.Tensor:
    """
    Brute force 2D grid for the same Hankel integral. 
    Integrates over x,y in the domain |α|≤1, Im(α)=y≤β.
    """
    p_coeffs = p_coeffs.to(torch.complex128)
    n = p_coeffs.numel() - 1
    xs = torch.linspace(-1.0, 1.0, Nxy, dtype=torch.float64)
    y_max = min(beta, 1.0)
    ys = torch.linspace(-1.0, y_max, Nxy, dtype=torch.float64)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    Z = torch.zeros((T, T), dtype=torch.complex128)

    def poly_p(z: complex) -> complex:
        deg = n
        res = 0j
        for k in range(deg + 1):
            c_k = p_coeffs[k]
            power = deg - k
            res += c_k * (z**power)
        return res

    for ix in range(Nxy):
        xv = xs[ix].item()
        for iy in range(Nxy):
            yv = ys[iy].item()
            if xv*xv + yv*yv <= 1.0:
                alpha = complex(xv, yv)
                val_p = poly_p(alpha)
                val_pc = poly_p(alpha.conjugate())
                fac = val_p * val_pc
                for i_idx in range(1, T + 1):
                    for j_idx in range(1, T + 1):
                        integrand = fac * (alpha**(i_idx - 1)) * ((alpha.conjugate())**(j_idx - 1))
                        Z[i_idx - 1, j_idx - 1] += integrand * dx * dy
    return Z

def accurate_zt_matrix(
    p_coeffs: torch.Tensor,
    T: int,
    beta: float,
    num_points: int = 601
) -> torch.Tensor:
    """More accurate implementation using proper complex integration"""
    p_coeffs = p_coeffs.to(torch.complex128)
    xs = torch.linspace(-1.0, 1.0, num_points, dtype=torch.float64)
    y_max = min(beta, 1.0)
    ys = torch.linspace(-1.0, y_max, num_points, dtype=torch.float64)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    Z = torch.zeros((T, T), dtype=torch.complex128)
    
    # Compute integral
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if x*x + y*y <= 1.0:
                alpha = complex(x.item(), y.item())
                val_p = eval_poly(alpha, p_coeffs)
                val_pc = eval_poly(alpha.conjugate(), p_coeffs)
                fac = val_p * val_pc
                
                for i in range(T):
                    for j in range(T):
                        Z[i, j] += fac * (alpha**i) * (alpha.conjugate()**j) * dx * dy
    
    return Z

def accurate_zt_matrix_vectorized(
    p_coeffs: torch.Tensor,
    T: int,
    beta: float,
    num_points: int = 601
) -> torch.Tensor:
    """Vectorized implementation using proper complex integration"""
    p_coeffs = p_coeffs.to(torch.complex128)
    
    # Create meshgrid for integration points
    xs = torch.linspace(-1.0, 1.0, num_points, dtype=torch.float64)
    y_max = min(beta, 1.0)
    ys = torch.linspace(-1.0, y_max, num_points, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    
    # Create mask for unit disk
    mask = (X**2 + Y**2 <= 1.0)
    
    # Create complex grid
    alpha = X[mask] + 1j * Y[mask]  # Shape: (N_valid_points,)
    
    # Evaluate polynomials for all points at once
    val_p = eval_poly(alpha, p_coeffs)
    val_pc = eval_poly(alpha.conj(), p_coeffs)
    fac = val_p * val_pc
    
    # Create powers matrix starting from 1..T instead of 0..(T-1)
    i_indices = torch.arange(1, T + 1, dtype=torch.float64).to(torch.complex128)
    powers_alpha = alpha.unsqueeze(1) ** (i_indices - 1)  # Shape: (N_valid_points, T)
    powers_alpha_conj = alpha.conj().unsqueeze(1) ** (i_indices - 1)
    
    # Compute final matrix
    Z = torch.zeros((T, T), dtype=torch.complex128)
    for k in range(len(alpha)):
        Z += fac[k] * torch.outer(powers_alpha[k], powers_alpha_conj[k])
    
    return Z * (dx * dy)

def eval_poly_vectorized(z: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Vectorized polynomial evaluation using Horner's method"""
    result = coeffs[-1].expand_as(z)
    for c in coeffs[-2::-1]:  # Iterate through coefficients in reverse order
        result = result * z + c
    return result

def compare_implementations(
    T: int = 4,
    beta: float = 0.5,
    num_points: int = 201,  # reduced for faster testing
    rtol: float = 1e-10,
    atol: float = 1e-10
) -> Tuple[bool, float, float, float]:
    """
    Compare the three implementations for correctness and performance.
    
    Returns:
        Tuple containing:
        - bool: Whether all implementations match within tolerance
        - float: Time taken by accurate_zt_matrix
        - float: Time taken by accurate_zt_matrix_vectorized
        - float: Time taken by hankel_brute
    """
    # Create some test polynomial coefficients
    p_coeffs = torch.tensor([1.0, -0.5, 0.1], dtype=torch.complex128)
    
    # Run and time each implementation
    def time_fn(fn: Callable) -> Tuple[torch.Tensor, float]:
        start = time.perf_counter()
        result = fn(p_coeffs, T, beta, num_points)
        end = time.perf_counter()
        return result, end - start

    Z1, t1 = time_fn(accurate_zt_matrix)
    Z2, t2 = time_fn(accurate_zt_matrix_vectorized)
    Z3, t3 = time_fn(lambda *args: hankel_brute(*args[:3], Nxy=args[3]))

    # Compare results
    match_1_2 = torch.allclose(Z1, Z2, rtol=rtol, atol=atol)
    match_1_3 = torch.allclose(Z1, Z3, rtol=rtol, atol=atol)
    match_2_3 = torch.allclose(Z2, Z3, rtol=rtol, atol=atol)
    all_match = match_1_2 and match_1_3 and match_2_3
    
    if not all_match:
        # Compute and print max differences
        print("\nMax absolute differences:")
        print(f"accurate vs vectorized: {torch.max(torch.abs(Z1 - Z2))}")
        print(f"accurate vs brute: {torch.max(torch.abs(Z1 - Z3))}")
        print(f"vectorized vs brute: {torch.max(torch.abs(Z2 - Z3))}")
        
        # Print a sample of matrices for debugging
        print("\nSample values from each matrix:")
        print("accurate_zt_matrix[0,0]:", Z1[0,0])
        print("vectorized[0,0]:", Z2[0,0])
        print("hankel_brute[0,0]:", Z3[0,0])
    
    return all_match, t1, t2, t3

def main():
    # Test cases with different sizes
    test_cases = [
        (4, 0.5, 201),   # Small case
        (8, 0.7, 301),   # Medium case
        (16, 0.9, 401),  # Larger case
    ]
    
    print("Running comparison tests...\n")
    
    for T, beta, num_points in test_cases:
        print(f"\nCase: T={T}, beta={beta}, num_points={num_points}")
        match, t1, t2, t3 = compare_implementations(T, beta, num_points)
        
        print(f"Results match: {match}")
        print("Timings:")
        print(f"  accurate_zt_matrix: {t1:.3f}s")
        print(f"  vectorized version: {t2:.3f}s")
        print(f"  hankel_brute:      {t3:.3f}s")
        print(f"Speedup (vectorized vs original): {t1/t2:.2f}x")
        print(f"Speedup (vectorized vs brute): {t3/t2:.2f}x")

if __name__ == "__main__":
    main()