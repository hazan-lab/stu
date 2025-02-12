import math

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel

try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False


class InitStdFactor(Enum):
    DISABLED = "disabled"           # factor = 1.0
    GLOBAL_DEPTH = "global_depth"   # factor = sqrt(2*(num_layers + 1))
    CURRENT_DEPTH = "current_depth" # factor = sqrt(2*(current_depth + 1))
    DIM_RATIO = "dim_ratio"         # factor = dim / 4096


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    This module implements the SwiGLU function defined as:
    FFN_SwiGLU(x, W, V, W2) = (Swish_{1}(xW) ⊙ (xV))W2
    where ⊙ denotes the Hadamard product and Swish_{1} is the Swish function with β=1.
    
    See more: https://arxiv.org/pdf/2002.05202

    Note: The Swish function with β=1 is equivalent to PyTorch's SiLU function.

    Args:
        dim (int): Input and output dimension.
        h_dim (int): Hidden dimension.
        bias (bool, optional): If false, additive biases will not be learned.
    
    Attributes:
         v (nn.Module): Additional linear layer for feature transformation.
         w (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
    """

    def __init__(self, dim: int, h_dim: int, bias: bool = False, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.w = nn.Linear(dim, h_dim, bias=bias, dtype=dtype)
        self.v = nn.Linear(dim, h_dim, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(h_dim, dim, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))

def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)

    return Z

def get_spectral_filters(
    seq_len: int, 
    K: int, 
    use_hankel_L: bool = False, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-14,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L)
    
    sigma, phi = np.linalg.eigh(Z, UPLO="U") 
    if not (sigma > -tol).all():
        raise ValueError(f"expected all eigenvalues >= 0 within a tolerance of {tol}, but found min = {sigma.min()}")
    
    sigma_k = sigma[-K:]
    phi_k = phi[:, -K:]

    phi_k = phi_k * (sigma_k ** 0.25)
    
    return torch.tensor(phi_k, device=device, dtype=dtype)

def compute_dimensions(n: int) -> tuple[int, int, int]:
    if n <= 2:
        raise ValueError("n must be greater than 2")

    T_prime = (math.ceil(math.sqrt(n - 2)))**2 + 2
    sqrt_T_prime = math.ceil(math.sqrt(T_prime - 2))
    k_max = sqrt_T_prime
    return T_prime, sqrt_T_prime, k_max

def get_tensorized_spectral_filters(
    n: int = 8192,
    k: int = 24,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute tensorized spectral filters for given sequence length and filter count.

    Args:
        n: Sequence length
        k: Number of filters
        use_hankel_L: Hankel_main ⊗ Hankel_L? Default is Hankel_main ⊗ Hankel_main.
        device: Computation device
        dtype: Computation dtype
    """
    assert torch.cuda.is_available(), "CUDA is required."

    T_prime, sqrt_T_prime, k_max = compute_dimensions(n)
    k = min(k, k_max)

    Z = get_hankel(sqrt_T_prime)
    sigma, phi = np.linalg.eigh(Z, UPLO="U") 
    phi_i = phi[:, -k:] * (sigma[-k:] ** 0.25)

    if use_hankel_L: # TODO: We may want to use Hankel_L above too if use_hankel_L is true, make another variable for this (mix != use_hankel_L)
        print("Mixing Hankel_L with Hankel_main to generate tensorized filters.")
        Z_L = get_hankel(sqrt_T_prime, True)
        sigma_L, phi_L = np.linalg.eigh(Z_L, UPLO="U") 
        phi_j = phi_L[:, -k:] * (sigma_L[-k:] ** 0.25)
    else:
        phi_j = phi_i

    phi_i, phi_j = torch.tensor(phi_i, device=device, dtype=dtype), torch.tensor(phi_j, device=device, dtype=dtype)
    filters = torch.kron(phi_i, phi_j)
    return filters

def get_optimal_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7/6) * math.log2(seq_len)))

def get_monic_chebyshev_coeffs(n: int) -> torch.Tensor:
    """
    TODO: Maybe this should be of 2nd kind, wait on what Elad says

    Get coefficients of monic n-th Chebyshev polynomial in descending order.
    Returns coefficients [c_n, c_{n-1}, ..., c_0] as complex128 tensor.
    """
    def chebyshev_t_int(n: int) -> list[int]:
        """Compute T_n coefficients exactly in integer arithmetic."""
        if n == 0:
            return [1]
        elif n == 1:
            return [1, 0]
        
        T0 = [1]       # T_0(x) = 1
        T1 = [1, 0]    # T_1(x) = x
        
        # Use recurrence T_n = 2xT_{n-1} - T_{n-2}
        for _ in range(2, n + 1):
            # Multiply T_{k-1} by 2x: scale by 2 and append 0
            T2 = [2*c for c in T1] + [0]
            
            # Subtract T_{k-2} with proper padding
            d = len(T2) - len(T0)
            padded_T0 = [0] * d + T0
            T2 = [a - b for a, b in zip(T2, padded_T0, strict=True)]
            
            T0, T1 = T1, T2
            
        return T2
    
    # Get standard Chebyshev coefficients and make monic
    coeffs = torch.tensor(chebyshev_t_int(n), dtype=torch.complex128)
    if n > 0:
        coeffs = coeffs / (2.0 ** (n - 1))
    return coeffs

def get_polynomial_hankel_matrix(
    seq_len: int,
    p_coeffs: torch.Tensor,
    beta: float = 1/64,
    num_points: int = 601
) -> torch.Tensor:
    """
    Compute complex Hankel matrix by integrating over complex plane.
    
    Args:
        seq_len: Length of sequence
        p_coeffs: Polynomial coefficients
        beta: Bound on imaginary components
        num_points: Number of quadrature points
        
    Returns:
        Complex Hankel matrix of shape (seq_len, seq_len)
    """
    p_coeffs = p_coeffs.to(torch.complex128)

    # Integration grid
    xs = torch.linspace(-1.0, 1.0, num_points, dtype=torch.float64)
    y_max = min(beta, 1.0)
    ys = torch.linspace(-1.0, y_max, num_points, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # Mask for unit disk
    mask = X**2 + Y**2 <= 1.0

    # Evaluate polynomial at complex points
    alpha = torch.complex(X[mask], Y[mask])
    
    def eval_poly(z: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomial using Horner's method.
        
        Args:
            z: Points to evaluate at
            coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_0]
        """
        result = coeffs[0].expand_as(z)
        for c in coeffs[1:]:
            result = result * z + c
        return result

    val_p = eval_poly(alpha, p_coeffs)
    val_pc = eval_poly(alpha.conj(), p_coeffs)
    fac = val_p * val_pc

    # Compute powers matrix 
    i_indices = torch.arange(seq_len, dtype=torch.float64).to(torch.complex128)
    powers_alpha = alpha.unsqueeze(1) ** i_indices
    powers_alpha_conj = alpha.conj().unsqueeze(1) ** i_indices

    # Final matrix via batch operations
    Z = torch.einsum("n,ni,nj->ij", fac, powers_alpha, powers_alpha_conj)

    return Z * (dx * dy)

def get_polynomial_spectral_filters(
    seq_len: int,
    k: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-14
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get spectral filters for asymmetric LDS via complex Hankel matrix.
    
    Args:
        seq_len: Length of sequence
        k: Number of filters to return
        p_coeffs: Polynomial coefficients 
        beta: Bound on imaginary components
        device: Output device
        dtype: Output dtype
        tol: Tolerance for eigenvalue positivity check
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) of shapes (k,) and (seq_len, k)
    """
    # Get optimal parameters from Theorem 2
    n = get_optimal_degree(seq_len)
    beta = 1.0 / (64 * n * n)
    
    # Get Chebyshev polynomial coefficients
    p_coeffs = get_monic_chebyshev_coeffs(n)
    
    # Compute complex Hankel matrix using equations (2) and (3)
    Z = get_polynomial_hankel_matrix(seq_len, p_coeffs, beta)
    
    # Get spectral filters via eigendecomposition
    sigma, phi = torch.linalg.eigh(Z, UPLO="U")

    # Verify eigenvalue positivity per paper
    if not (sigma > -tol).all():
        raise ValueError(f"Expected positive eigenvalues, got min={sigma.min()}")

    # Take top k eigenpairs
    sigma_k = sigma[-k:].to(device=device, dtype=dtype)  # TODO: Should we do the ** 0.25 thing here?
    phi_k = phi[:, -k:].to(device=device, dtype=dtype)

    return phi_k.to(device=device, dtype=dtype)

def conv(u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        u (torch.Tensor): Input tensor of shape (B, L, D)
        v (torch.Tensor): Filter tensor of shape (K, D)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple `(U_plus, U_minus)`:
            - `U_plus`: Convolved output tensor with positive eigenvalues.
            - `U_minus`: Convolved output tensor with negative eigenvalues.
            - Shapes of the output tensors depend on `use_tensordot`:
                - If `use_tensordot=True`: `(B, L, D)`
                - If `use_tensordot=False`: `(B, L, K, D)`
    """
    bsz, seq_len, dim = u.shape
    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    if use_tensordot:
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32) # (bsz, seq_len, K, dim, stack)
        u = u.view(bsz, -1, 1, dim).expand(bsz, -1, K, dim)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus.to(dtype=u.dtype), U_minus.to(dtype=u.dtype)


def flash_conv(
    u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_tensordot: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flash FFT convolution.

    Args:
        u (torch.Tensor): Input tensor of shape `(B, L, d_in)`, where:
            - `B` is the batch size,
            - `L` is the sequence length,
            - `d_in` is the input dimension.
        v (torch.Tensor): Filter tensor of shape `(K, d_in)`, where:
            - `K` is the number of filters,
            - `d_in` is the input dimension.
        flash_fft (FlashFFTConv): An instance of the FlashFFTConv module, used to perform the convolution.
        use_tensordot (bool, optional): If `True`, performs the tensordot approximation (default is `True`).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple `(U_plus, U_minus)`:
            - `U_plus`: Convolved output tensor with positive eigenvalues.
            - Shape depends on `use_tensordot`:
                - If `use_tensordot=True`: `(B, L, d_in)`
                - If `use_tensordot=False`: `(B, L, K, d_in)`
            - `U_minus`: Convolved output tensor with negative eigenvalues.
            - Shape depends on `use_tensordot`:
                - If `use_tensordot=True`: `(B, L, d_in)`
                - If `use_tensordot=False`: `(B, L, K, d_in)`

    Raises:
        ValueError: If the input tensor shapes do not conform to the expected dimensions.

    Example:
        >>> u = torch.randn(4, 16, 32)  # (B, L, d_in)
        >>> v = torch.randn(8, 32)      # (K, d_in)
        >>> flash_fft = FlashFFTConv(n=16, dtype=torch.float32)
        >>> U_plus, U_minus = flash_conv(u, v, flash_fft, use_tensordot=True)
        >>> print(U_plus.shape, U_minus.shape)
        torch.Size([4, 16, 32]) torch.Size([4, 16, 32])
        """
    bsz, seq_len, dim = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    sgn = torch.full((1, 1, padded_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1

    if use_tensordot:
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, dim, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(dim, 1).contiguous()
        u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * dim, padded_len)

    U_conv = flash_fft(u_conv, v_padded)
    U_conv = U_conv[..., :seq_len] # Trim output back to original sequence length

    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)
    if use_tensordot:
        u_minus = u_minus * sgn[:, :, :seq_len]
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, dim, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, dim, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

    return U_plus.to(dtype=u.dtype), U_minus.to(dtype=u.dtype)


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Rolls the time axis forward by k steps to align the input u_{t-k} with u_t,
    removing the last k time steps of the input tensor.

    Args:
        u (torch.Tensor): An input tensor of shape [B, L, K, D].
        k (int): Number of time steps to shift. Defaults to 1.

    Returns:
        torch.Tensor: Shifted tensor of shape [B, L, K, D].
    """
    if k == 0:
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted

def compute_ar_y_teacher_forcing(M_y: torch.Tensor, y_ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Parallel teacher-forcing approach for AR over outputs.
    M_y: (dim, k_y, dim)
    y_ground_truth: (B, L, dim)  # the *true* or gold outputs
    Returns: (B, L, dim)
    """
    k_y = M_y.shape[1]
    y_shifted = torch.stack([shift(y_ground_truth, i) for i in range(1, k_y+1)], dim=1)  # (B,k,L,dim)
    ar_y = torch.einsum("bkld,dki->bli", y_shifted, M_y)
    return ar_y

def compute_ar_u(M_u: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Computes the autoregressive component of the STU model with respect to
    the input, as described in Equation (4) of Section 3.

    This function implements the sum of M^u_i u_{t+1-i} from i=1 to
    (more generally) k_u (in the paper, it was up until i=3).

    Args:
        M_u (torch.Tensor): Input weight matrices of shape (d_in, k_u, d_out)
        u (torch.Tensor): Input tensor of shape (B, L, d_in)

    Returns:
        torch.Tensor: Autoregressive component w.r.t. input of shape (B, L, d_out)
    """
    # TODO: Use Tri Dao's causal conv1d instead (check if cuda available) else stacking+einsum
    k_u = M_u.size(1)
    u_shifted = torch.stack([shift(u, i) for i in range(1, k_u+1)], dim=1)
    ar_u = torch.einsum("bkld,dki->bli", u_shifted, M_u)
    return ar_u

def compute_ar_strict(M_y: torch.Tensor, spectral: torch.Tensor, u: torch.Tensor, M_u: torch.Tensor) -> torch.Tensor:
    """
    A naive step-by-step approach for AR over the inputs and the model’s own outputs.
    """
    bsz, seq_len, dim = spectral.shape
    device = spectral.device
    # M_y is (dim, k_y, dim)
    # M_u is (dim, k_u, dim) if not None
    _, k_y, _ = M_y.shape
    k_u = M_u.shape[1] if M_u is not None else 0

    # We'll store the final outputs in 'out'
    out = torch.zeros_like(spectral)

    # Rolling buffer for the last k_y *outputs*:
    #   y_buffer[:, 0, :] => y_{t-1}
    #   y_buffer[:, 1, :] => y_{t-2}, etc.
    y_buffer = torch.zeros(bsz, k_y, dim, device=device, dtype=spectral.dtype)

    for t in range(seq_len):

        # 1) Autoregressive input sum: sum_{j=1..k_u}( M^u_j * u_{t+1-j} )
        if M_u is not None:
            y_in = torch.zeros(bsz, dim, device=device, dtype=spectral.dtype)
            for j in range(1, k_u+1):
                t_in = (t + 1) - j
                if 0 <= t_in < seq_len:
                    # (bsz, d_in) @ (d_in, dim) => (bsz, dim)
                    y_in += u[:, t_in] @ M_u[:, j-1, :].T
        else:
            y_in = 0

        # 2) Autoregressive output sum: sum_{j=1..k_y}( M^y_j * y_out(t-j) )
        #    stored in y_buffer => shape (bsz, k_y, dim)
        #    M_y                => shape (dim, k_y, dim)
        # We can do it in one fused operation with einsum("b j i, d j i -> b d").
        y_out_ar = torch.einsum("bjf, djf->bd", y_buffer, M_y)

        # 3) Add the spectral term for time t: spectral[:, t, :]
        spec_t = spectral[:, t, :]

        # 4) Combine them all: y_t = spectral + input-AR + output-AR
        y_t = spec_t + y_in + y_out_ar

        # Save y_t as the output for time t
        out[:, t, :] = y_t

        # 5) Shift the rolling buffer to the right, so that the new y_t
        #    will appear as y_{t-1} in the next iteration:
        #
        #  - The old y_buffer[:,0,:] becomes y_buffer[:,1,:]
        #  - The old y_buffer[:,1,:] becomes y_buffer[:,2,:], etc.
        #  - Then we insert y_t at y_buffer[:,0,:].
        y_buffer = torch.roll(y_buffer, shifts=1, dims=1)
        y_buffer[:, 0, :] = y_t

    return out

class STU(nn.Module):
    def __init__(self, config, filters, n) -> None:
        """
        STU module combining convolution with autoregressive (AR) components.

        Args:
            config: Configuration with attributes:
                - d_in (int): Input dimension.
                - dim (int): Model dimension.
                - d_out (int): Output dimension.
                - seq_len (int): Sequence length.
                - k_y (int): Number of AR components for outputs.
                - k_u (int): Number of AR components for inputs.
                - num_filters (int): Number of spectral filters.
                - use_hankel_L (bool): Whether to use the alternative Hankel matrix.
                - use_tensordot (bool): Whether to use the tensordot approximation.
                - learnable_M_y (bool): Whether the M_y matrix is learnable.
            filters: Spectral filters.
            n (int): FFT length (typically nearest_power_of_two(seq_len*2-1)).
        """
        super(STU, self).__init__()
        assert config.k_y < n, "k_y must be less than the sequence length"
        assert config.k_u < n, "k_u must be less than the sequence length"

        self.config = config
        self.d_in = config.d_in
        self.dim = config.dim
        self.d_out = config.d_out
        self.filters = filters
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.flash_fft = ( # TODO: Buggy with torch.compile, need to write a custom op wrapper
            FlashFFTConv(self.n, dtype=torch.bfloat16)
            if config.use_flash_fft and flash_fft_available
            else None
        )
        self.num_filters = config.num_filters
        self.use_hankel_L = config.use_hankel_L
        self.use_tensordot = config.use_tensordot  # Tensordot approx
        self.learnable_M_y = config.learnable_M_y
        self.k_u = config.k_u
        self.k_y = config.k_y
        self.use_teacher_forcing = config.use_teacher_forcing
        self.use_polynomial_spectral_filters = config.use_polynomial_spectral_filters
        self.use_norm = config.use_norm

        # Parameterizable matrix Mᵘ, Mᵠ⁺, and Mᵠ⁻, per section 3
        self.M_u = nn.Parameter(torch.zeros(self.dim, self.k_u, self.dim))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_M_y:
            self.M_y = nn.Parameter(torch.zeros(self.dim, self.k_y, self.dim))
        else:
            identity = torch.zeros(self.dim, self.dim, self.dim)
            self.register_buffer("M_y", identity)

        if self.use_tensordot:
            self.M_inputs = nn.Parameter(torch.zeros(self.dim, self.dim))
            self.M_filters = nn.Parameter(torch.zeros(self.num_filters, self.dim))
        else:
            self.M_phi_plus = nn.Parameter(torch.zeros(self.dim, self.num_filters, self.dim))
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.zeros(self.dim, self.num_filters, self.dim))

    def forward(self, u: torch.Tensor, y_ground_truth: torch.Tensor = None) -> torch.Tensor:
        if self.use_tensordot:
            # Contract inputs and filters over the K and model dims, then convolve
            u_proj = u @ self.M_inputs
            phi_proj = self.filters @ self.M_filters

            if self.flash_fft:
                spectral_plus, spectral_minus = flash_conv(u_proj, phi_proj, self.flash_fft, self.use_tensordot)
            else:
                spectral_plus, spectral_minus = conv(u_proj, phi_proj, self.n, self.use_tensordot)
        else:
            # Convolve inputs and filters,
            if self.flash_fft:
                U_plus, U_minus = flash_conv(u, self.filters, self.flash_fft, self.use_tensordot)
            else:
                U_plus, U_minus = conv(u, self.filters, self.n, self.use_tensordot)

            U_plus_shifted = shift(U_plus, 2)
            if not self.use_hankel_L:
                U_minus_shifted = shift(U_minus, 2)

            # Then, contract over the K and model dims
            U_plus_shifted = U_plus_shifted.to(self.M_phi_plus.dtype)
            spectral_plus = torch.einsum("blki,dki->bli", U_plus_shifted, self.M_phi_plus)
            if not self.use_hankel_L:
                U_minus_shifted = U_minus_shifted.to(self.M_phi_plus.dtype)
                spectral_minus = torch.einsum("blki,dki->bli", U_minus_shifted, self.M_phi_minus)

        spectral = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

        if self.k_y == 0:
            if self.k_u > 0:
                ar_in = compute_ar_u(self.M_u, u)
                return spectral + ar_in
            else:
                return spectral

        if self.use_teacher_forcing:
            if y_ground_truth is None:
                raise ValueError("Teacher forcing requires y_ground_truth to be provided.")
            ar_out = compute_ar_y_teacher_forcing(self.M_y, y_ground_truth)
            ar_in = compute_ar_u(self.M_u, u) if (self.k_u > 0) else 0
            return spectral + ar_in + ar_out

        if self.k_y > 0:
            if self.use_teacher_forcing:
                if y_ground_truth is None:
                    raise ValueError(f"Teacher forcing requires y_ground_truth, got {y_ground_truth}.")
                ar_out = compute_ar_y_teacher_forcing(self.M_y, y_ground_truth)
                if self.k_u > 0:
                    ar_in = compute_ar_u(self.M_u, u)
                final = ar_out + ar_in + spectral
            else:
                final = compute_ar_strict(self.M_y, spectral, u, self.M_u)
            return final
        else:
            # Strict step-by-step for both output AR and input AR
            return compute_ar_strict(self.M_y, spectral, u, self.M_u)


class STULayer(nn.Module):
    """
    An STU (Spectral Transform Unit) layer.
    """

    def __init__(self, config, phi, n) -> None:
        super(STULayer, self).__init__()
        self.stu = STU(config, phi, n)
        self.stu_norm = nn.RMSNorm(config.dim, dtype=config.torch_dtype) if config.use_norm else nn.Identity()
        self.use_mlp = config.use_mlp
        self.dropout = nn.Dropout(p=config.dropout)

        if self.use_mlp:
            self.mlp = MLP(dim=config.dim, h_dim=config.mlp_scale * config.dim, bias=config.bias, dtype=config.torch_dtype)
            self.mlp_norm = nn.RMSNorm(config.dim, dtype=config.torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STU layer.

        Args:
            x (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, d_out)
        """
        x = x + self.dropout(self.stu(self.stu_norm(x)))
        if self.use_mlp:
            x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class SpectralSSMConfig(PretrainedConfig):
    model_type = "spectral_ssm"

    def __init__(
        self,
        bsz: int = 1,
        num_layers: int = 2,
        d_in: int = 32,
        dim: int = 32,
        d_out: int = 32,
        seq_len: int = 1024,
        bias: bool = False,
        num_filters: int = 24,
        k_y: int = 2,
        k_u: int = 3,
        learnable_M_y: bool = True,
        use_mlp: bool = True,
        use_tensordot: bool = False,
        use_hankel_L: bool = False,
        use_flash_fft: bool = False,
        use_teacher_forcing: bool = False,
        use_polynomial_spectral_filters: bool = False,
        use_norm: bool = True,
        mlp_scale: int = 4,
        dropout: float = 0.1,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        **kwargs,
    ):
        # Call the parent constructor with any additional kwargs.
        super().__init__(**kwargs)
        
        # Model structural parameters
        self.bsz = bsz
        self.num_layers = num_layers
        self.d_in = d_in
        self.dim = dim
        self.d_out = d_out
        self.seq_len = seq_len
        self.bias = bias
        self.num_filters = num_filters
        
        # AR-related parameters
        self.k_y = k_y
        self.k_u = k_u
        self.learnable_M_y = learnable_M_y

        # STU customization flags
        self.use_tensordot = use_tensordot
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_teacher_forcing = use_teacher_forcing
        self.use_polynomial_spectral_filters = use_polynomial_spectral_filters
        self.use_norm = use_norm
        self.use_mlp = use_mlp
        self.mlp_scale = mlp_scale
        self.dropout = dropout

        # Device and dtype settings
        self.torch_dtype = torch_dtype
        self.device = device


class SpectralSSM(PreTrainedModel):
    config_class = SpectralSSMConfig

    def __init__(self, config, phi):
        super(SpectralSSM, self).__init__(config)
        self.config = config
        self.d_in = config.d_in
        self.dim = config.dim
        self.d_out = config.d_out
        self.filters = phi  # Precomputed FFT of top K eigenvectors.
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.num_layers = config.num_layers
        self.use_tensordot = config.use_tensordot
        self.use_hankel_L = config.use_hankel_L
        self.k_u = config.k_u
        self.k_y = config.k_y
        self.init_std_factor = InitStdFactor.CURRENT_DEPTH
        self.init_base_std = 0.02
        self.std = self.init_base_std

        if config.k_y > 0 and not config.use_teacher_forcing:
            print("\nWARNING: k_y > 0 and use_teacher_forcing=False. "
                  "(!!) The STU model will be performing true autoregressive prediction over its own outputs. (!!) "
                  "This is the strongest mode in theory, but it is not recommended in practice as it will lead to extremely degraded efficiency.\n"
                )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(STULayer(config, self.filters, self.n))

        self.in_proj = nn.Linear(config.d_in, config.dim, bias=config.bias, dtype=config.torch_dtype)
        self.norm = nn.RMSNorm(config.dim, dtype=config.torch_dtype) if config.use_norm else nn.Identity()
        self.out_proj = nn.Linear(config.dim, config.d_out, bias=config.bias, dtype=config.torch_dtype)

        # First, perform a general initialization, and zero out all M_ matrices
        self.apply(self._init_weights)
        # Then, do our by-layer init scheme for layers in self.layers:
        self._init_weights_by_layer()

        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def _get_num_params(self):
        num_params = sum(param.numel() for param in self.parameters())
        return num_params

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)

        # Explicitly zero out parameters starting with "M_", per the paper.
        for name, param in module.named_parameters(recurse=False):
            if name.startswith("M_"):
                with torch.no_grad():
                    param.zero_()
    
    def _init_weights_by_layer(self):
        """
        For each layer in self.layers, compute a scaling factor based on the chosen strategy,
        and reinitialize any nn.Linear submodules in that layer with a scaled std.
        """
        num_layers = len(self.layers)
        for depth, layer in enumerate(self.layers):
            # Compute factor based on our chosen strategy:
            if self.init_std_factor == InitStdFactor.CURRENT_DEPTH:
                factor = math.sqrt(2 * (depth + 1))
            elif self.init_std_factor == InitStdFactor.GLOBAL_DEPTH:
                factor = math.sqrt(2 * (num_layers + 1))
            elif self.init_std_factor == InitStdFactor.DIM_RATIO:
                factor = self.dim / 4096.0
            elif self.init_std_factor == InitStdFactor.DISABLED:
                factor = 1.0
            else:
                factor = 1.0

            # for every nn.Linear in the current layer, reinitialize with the new std:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_base_std * factor)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward(self, u):
        u = self.in_proj(u)

        for layer in self.layers:
            u = layer(u)

        y_hat = self.out_proj(self.norm(u))

        return y_hat

if __name__ == "__main__":
    import json
    config_path = "config.json"

    with open(config_path, "r") as f:
        config_data = json.load(f)

    # set device based on availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # MPS support for PyTorch still highly sus
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # get torch dtype from the config; config key now is "dtype"
    torch_dtype = getattr(torch, config_data["torch_dtype"])

    # build the model configuration
    configs = SpectralSSMConfig(
        num_layers=config_data["num_layers"],
        d_in=config_data["d_in"],
        dim=config_data["dim"],
        d_out=config_data["d_out"],
        seq_len=config_data["seq_len"],
        bias=config_data["bias"],
        num_filters=config_data["num_filters"] or math.ceil(math.log(config_data["seq_len"])),
        k_y=config_data["k_y"],
        k_u=config_data["k_u"],
        learnable_M_y=config_data["learnable_M_y"],
        use_mlp=config_data["use_mlp"],
        use_tensordot=config_data["use_tensordot"],
        use_hankel_L=config_data["use_hankel_L"],
        use_flash_fft=config_data["use_flash_fft"],
        use_teacher_forcing=config_data["use_teacher_forcing"],
        use_norm=config_data["use_norm"],
        dtype=torch_dtype,
        device=device,
    )

    filters = get_spectral_filters(
        seq_len=config_data["seq_len"],
        K=config_data["num_filters"] or math.ceil(math.log(config_data["seq_len"])),
        use_hankel_L=config_data["use_hankel_L"],
        device=device,
        dtype=torch_dtype,
    )

    print("Configs:")
    for key, value in vars(configs).items():
        print(f"  {key}: {value}")

    model = SpectralSSM(configs, filters).to(device=device, dtype=torch_dtype)
    x = torch.randn(configs.bsz, configs.seq_len, configs.d_in, device=device, dtype=torch_dtype)
    outputs = model(x)

    print("Output shape:", outputs.shape)
    print("Sample output:", outputs[0, 0, :10])
