import math
import torch

def _simpson_1d(values: torch.Tensor, dx: float) -> float:
    """
    Basic Simpson's rule for evenly spaced 1D samples.
    If len(values) is even, drops the last sample.
    """
    n = values.shape[0]
    if n < 2:
        return 0.0
    if n % 2 == 0:
        n -= 1
        values = values[:n]
    s = values[0] + values[-1]
    s += 4.0 * torch.sum(values[1:-1:2])
    s += 2.0 * torch.sum(values[2:-1:2])
    return (dx / 3.0) * s.item()

def hankel_polar(
    p_coeffs: torch.Tensor,
    T: int,
    beta: float,
    num_theta_points: int = 511,
    num_r_points: int = 511
) -> torch.Tensor:
    """
    Single-polar integration of the Hankel matrix:
       Z_{i,j} = ∫_{|α|≤1, Im(α)≤β} p(α)p(α̅) α^(i−1) α̅^(j−1) dα.
    Uses r,θ with r in [0, Rmax(θ)] and θ in [−π, π].
    """
    p_coeffs = p_coeffs.to(torch.complex128)
    n = p_coeffs.numel() - 1
    Z = torch.zeros((T, T), dtype=torch.complex128)

    def poly_p(z: complex) -> complex:
        deg = n
        res = 0j
        for k in range(deg + 1):
            c_k = p_coeffs[k]
            power = deg - k
            res += c_k * (z**power)
        return res

    thetas = torch.linspace(-math.pi, math.pi, num_theta_points, dtype=torch.float64)
    dtheta = thetas[1] - thetas[0]
    partial_list = []

    for th_i in range(num_theta_points):
        th = thetas[th_i].item()
        s_th = math.sin(th)
        if s_th > 0:
            rmax = min(1.0, beta / s_th)
            if rmax <= 1e-15:
                partial_list.append(torch.zeros((T, T), dtype=torch.complex128))
                continue
        else:
            rmax = 1.0
        rs = torch.linspace(0.0, rmax, num_r_points, dtype=torch.float64)
        dr = rs[1] - rs[0] if num_r_points > 1 else 0.0
        fvals = torch.zeros((num_r_points, T, T), dtype=torch.complex128)
        for ri in range(num_r_points):
            rv = rs[ri].item()
            alpha = rv * complex(math.cos(th), math.sin(th))
            val_p = poly_p(alpha)
            val_pc = poly_p(alpha.conjugate())
            fac = val_p * val_pc
            apows = [alpha**k for k in range(T)]
            cpows = [(alpha.conjugate())**k for k in range(T)]
            for i_idx in range(T):
                for j_idx in range(T):
                    fvals[ri, i_idx, j_idx] = fac * apows[i_idx] * cpows[j_idx]
            fvals[ri] *= rv
        accum_mat = torch.zeros((T, T), dtype=torch.complex128)
        for i_idx in range(T):
            for j_idx in range(T):
                accum_mat[i_idx, j_idx] = _simpson_1d(fvals[:, i_idx, j_idx], dr)
        partial_list.append(accum_mat)

    for i_idx in range(T):
        for j_idx in range(T):
            arr = torch.tensor([pmat[i_idx, j_idx] for pmat in partial_list], dtype=torch.complex128)
            Z[i_idx, j_idx] = _simpson_1d(arr, dtheta.item())

    return Z


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


if __name__ == "__main__":
    # Example usage
    p = torch.tensor([1.0, 0.3, -0.2], dtype=torch.float64)  # x^2 + 0.3x - 0.2
    Tdim = 3
    b = 0.5

    Z_polar = hankel_polar(p, Tdim, b, num_theta_points=1001, num_r_points=1001)
    Z_brute = hankel_brute(p, Tdim, b, Nxy=601)

    diff = (Z_polar - Z_brute).abs()
    rel_err = diff.norm() / Z_brute.norm()

    print("Polar Hankel (real part):\n", Z_polar.real)
    print("\nBrute Force (real part):\n", Z_brute.real)
    print(f"\nRel error: {rel_err.item():.3e}")
