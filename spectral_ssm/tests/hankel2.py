import numpy as np
import torch
from typing import Tuple
import matplotlib.pyplot as plt

def simple_zt_matrix(T: int, beta: float, c1: float) -> np.ndarray:
    """Simple implementation assuming p(α) = α + c1"""
    i = np.arange(1, T + 1)[:, None]  # column vector
    j = np.arange(1, T + 1)[None, :]  # row vector
    
    denom1 = i + j + 1
    denom2 = i + j
    denom3 = np.maximum(i + j - 1, 1)
    
    term1 = 1.0 / denom1
    term2 = 2 * c1 / denom2
    term3 = (c1 * c1) / denom3
    
    arcsin_beta = np.arcsin(min(beta, 1.0))
    return 2 * arcsin_beta * (term1 + term2 + term3)

def eval_poly(z: complex, coeffs: torch.Tensor) -> complex:
    """Evaluate polynomial with given coefficients at point z"""
    result = complex(0, 0)
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        result += c.item() * (z ** power)
    return result

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

def compare_methods(T: int, beta: float, plot: bool = True) -> Tuple[np.ndarray, torch.Tensor]:
    """Compare simple and accurate implementations"""
    # Simple method parameters
    c1 = 0.3
    Z_simple = simple_zt_matrix(T, beta, c1)
    
    # Accurate method parameters
    p_coeffs = torch.tensor([1.0, c1], dtype=torch.float64)  # α + c1
    Z_accurate = accurate_zt_matrix(p_coeffs, T, beta)
    
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot simple method
        im1 = ax1.imshow(Z_simple.real)
        ax1.set_title('Simple Method')
        plt.colorbar(im1, ax=ax1)
        
        # Plot accurate method
        im2 = ax2.imshow(Z_accurate.real)
        ax2.set_title('Accurate Method')
        plt.colorbar(im2, ax=ax2)
        
        # Plot difference
        diff = np.abs(Z_simple - Z_accurate.numpy().real)
        im3 = ax3.imshow(diff)
        ax3.set_title('Absolute Difference')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()
    
    return Z_simple, Z_accurate

if __name__ == "__main__":
    # Test parameters
    T = 5
    beta = 0.5
    
    # Compare methods
    Z_simple, Z_accurate = compare_methods(T, beta)
    
    # Print some statistics
    rel_error = np.linalg.norm(Z_simple - Z_accurate.numpy().real) / np.linalg.norm(Z_accurate.numpy().real)
    print(f"\nRelative Error: {rel_error:.3e}")
    
    # Print eigenvalue stats
    eig_simple = np.linalg.eigvals(Z_simple)
    eig_accurate = np.linalg.eigvals(Z_accurate.numpy().real)
    
    print("\nLargest eigenvalues:")
    print(f"Simple:   {sorted(abs(eig_simple))[-3:]}")
    print(f"Accurate: {sorted(abs(eig_accurate))[-3:]}")