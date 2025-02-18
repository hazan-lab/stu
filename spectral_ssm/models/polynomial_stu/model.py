import math

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel

from spectral_ssm.models.polynomial_stu.cheby import (
    get_polynomial_hankel,
    normalized_chebyshev_coeff,
)

try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False


class InitStdFactor(Enum):
    DISABLED = "disabled"  # factor = 1.0
    GLOBAL_DEPTH = "global_depth"  # factor = sqrt(2*(num_layers + 1))
    CURRENT_DEPTH = "current_depth"  # factor = sqrt(2*(current_depth + 1))
    DIM_RATIO = "dim_ratio"  # factor = dim / 4096


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )


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

    def __init__(
        self,
        dim: int,
        h_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.w = nn.Linear(dim, h_dim, bias=bias, dtype=dtype)
        self.v = nn.Linear(dim, h_dim, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(h_dim, dim, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))


def get_opt_degree(seq_len: int) -> int:
    """
    Get optimal polynomial degree per Theorem 2: n = (7/6)log_2(T).
    """
    return int(math.ceil((7 / 6) * math.log2(seq_len)))


def get_polynomial_spectral_filters(
    seq_len: int,
    k: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    n = get_opt_degree(seq_len)
    beta = 1.0 / (64.0 * n**2)

    Z = get_polynomial_hankel(n, beta, seq_len, device=device)
    _, phi = torch.linalg.eigh(Z, UPLO="U")
    phi_k = phi[:, -k:] / math.sqrt(seq_len)

    # Validate that the eigenvectors are real since Z is Hermitian
    if torch.abs(phi_k.imag).max() > 1e-7:
        raise ValueError("Unexpectedly large imaginary components in eigenvectors")

    # Take real part only (imaginary part is due to floating point imprecision)
    return phi_k.real.to(dtype)


def conv(
    u: torch.Tensor, v: torch.Tensor, n: int, use_tensordot: bool = True
) -> torch.Tensor:
    """
    Args:
        u (torch.Tensor): Input tensor of shape (B, L, d_in)
        v (torch.Tensor): Filter tensor of shape (K, D)
        n (int): Length to pad to for FFT
        use_tensordot (bool): Whether to use tensordot approximation

    Returns:
        torch.Tensor: The convolved output tensor U_plus
    """
    bsz, seq_len, dim = u.shape

    if use_tensordot:
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        v = v.view(1, -1, K, 1).to(torch.float32)  # (bsz, seq_len, K, dim)
        u = u.view(bsz, -1, 1, dim).expand(bsz, -1, K, dim)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.fft.rfft(u.to(torch.float32), n=n, dim=1)
    U_plus = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]

    return U_plus.to(dtype=u.dtype)


def flash_conv(
    u: torch.Tensor,
    v: torch.Tensor,
    flash_fft: FlashFFTConv,
    use_tensordot: bool = True,
) -> torch.Tensor:
    """
    Flash FFT convolution optimized for polynomial STU.

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
        torch.Tensor: The convolved output tensor U_plus with shape:
            - If `use_tensordot=True`: `(B, L, d_in)`
            - If `use_tensordot=False`: `(B, L, K, d_in)`

    Example:
        >>> u = torch.randn(4, 16, 32)  # (B, L, d_in)
        >>> v = torch.randn(8, 32)      # (K, d_in)
        >>> flash_fft = FlashFFTConv(n=16, dtype=torch.float32)
        >>> U_plus = flash_conv(u, v, flash_fft, use_tensordot=True)
        >>> print(U_plus.shape)
        torch.Size([4, 16, 32])
    """
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    if use_tensordot:
        u_padded = (
            F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        )
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = u_padded.reshape(bsz, d_in, padded_len)
    else:
        u_k_padded = (
            F.pad(u.transpose(1, 2), (0, pad_len))
            .to(torch.bfloat16)
            .repeat_interleave(K, dim=1)
            .contiguous()
        )
        v_padded = (
            F.pad(v.transpose(0, 1), (0, pad_len))
            .to(torch.float32)
            .repeat(d_in, 1)
            .contiguous()
        )
        u_conv = u_k_padded.reshape(bsz, K * d_in, padded_len)

    U_conv = flash_fft(u_conv, v_padded)[..., :seq_len]

    if use_tensordot:
        U_plus = U_conv.transpose(1, 2)
    else:
        U_plus = U_conv.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()

    return U_plus


class STU(nn.Module):
    def __init__(self, config, filters, n) -> None:
        super().__init__()
        self.dim = config.dim
        self.num_filters = config.num_filters
        self.use_tensordot = config.use_tensordot
        self.filters = filters
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.num_terms = get_opt_degree(config.seq_len)
        p_coeffs = torch.tensor(
            normalized_chebyshev_coeff(self.num_terms-1),
            dtype=config.torch_dtype,
        )
        self.register_buffer("p_coeffs", p_coeffs)
        self.flash_fft = (  # TODO: Buggy with torch.compile, need to write a custom op wrapper
            FlashFFTConv(self.n, dtype=torch.bfloat16)
            if config.use_flash_fft and flash_fft_available
            else None
        )

        # NOTE: Learnable param with self.n and NOT a buffer; could be a memory bottleneck for large models.
        self.Q = nn.Parameter(torch.zeros(self.num_terms, self.dim, self.dim))

        if self.use_tensordot:
            self.M_proj = nn.Parameter(
                torch.zeros(self.num_filters, self.dim), dtype=config.torch_dtype
            )
            self.M_filters = nn.Parameter(
                torch.zeros(self.num_filters, self.dim, dtype=config.torch_dtype)
            )
        else:
            self.M = nn.Parameter(torch.zeros(self.dim, self.num_filters, self.dim))

    def _univariate_cheb_expansion(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the term -sum_{i=1}^n c_i * y_{t-i} via FFT convolution.
        """
        bsz, seq_len, d_in = u.shape
        v = self.p_coeffs.flip(0)  # For true conv
        v = v.unsqueeze(0)  # (1, n)
        v = v.expand(d_in, -1)  # (d_in, n)

        V = torch.fft.rfft(v.to(torch.float32), n=self.n, dim=1)
        V = V.transpose(0, 1)  # (F, d_in)
        U = torch.fft.rfft(u.to(torch.float32), n=self.n, dim=1)
        U_conv = torch.fft.irfft(V * U, n=self.n, dim=1)[:, :seq_len]

        return -U_conv.to(u.dtype)

    def _bivariate_cheb_expansion(self, u: torch.Tensor) -> torch.Tensor:
        """
        Efficiently computes the term:

          out[b, t, :] = sum_{s=0}^{n-1} sum_{i=1}^n c[i] * Q[max(0, i-n + s)] @ u[b, t-s, :]

        in a vectorized manner. See tests/bivariate_cheb_expansion.py.
        """
        B, L, d_in = u.shape
        n, _, d_out = self.Q.shape
        assert self.p_coeffs.shape[0] == n, (
            "Mismatch: p_coeffs and Q must share 'n' dimension"
        )

        # 1) Build a stack of shifted versions of u for s in [0..n-1]
        T = torch.arange(L, device=u.device)
        S = torch.arange(n, device=u.device)
        ts_idx = T.unsqueeze(0) - S.unsqueeze(1)  # (n, L)
        ts_idx_clamped = ts_idx.clamp(min=0, max=L - 1)  # (n, L)

        u_shifts = u[:, ts_idx_clamped, :]  # (B, n, L, d_in)
        u_shifts = u_shifts.permute(1, 0, 2, 3)  # (n, B, L, d_in)

        # Zero out positions where t-s < 0
        mask = (ts_idx < 0).unsqueeze(1).unsqueeze(-1)  # shape (n,1,L,1)
        u_shifts = u_shifts.masked_fill(mask.expand_as(u_shifts), 0)

        # 2) For i in [1..n], s in [0..n-1], pick Q[max(0, i-n+s)]
        i_idx = torch.arange(1, n + 1, device=u.device).view(n, 1)  # shape (n,1)
        s_idx = torch.arange(n, device=u.device).view(1, n)  # shape (1,n)
        offset = (i_idx - n + s_idx).clamp(min=0, max=n - 1)  # shape (n,n)
        Q_offset = self.Q[offset]  # shape (n,n,d_in,d_out)

        # 3) Combine with c[i] in an einsum
        Z = torch.einsum("isrd,sbtr->isbtd", Q_offset, u_shifts)  # (n,n,B,L,d_out)
        Z = Z * self.p_coeffs.view(n, 1, 1, 1, 1)  # broadcast c[i]
        out = Z.sum(dim=(0, 1))  # sum over i,s => (B,L,d_out)
        return out

    def _spectral_component(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the spectral component of Algorithm 1.

        Args:
            u (torch.Tensor): The input tensor of shape (B, L, D).

        Returns:
            torch.Tensor: The spectral component of shape (B, L, D).
        """
        if self.use_tensordot:
            # 1) Project input and filters, then convolve
            u_proj = u @ self.M_proj  # shape (B,L,num_filters)
            phi_proj = self.filters @ self.M_proj  # shape (K, num_filters)
            if self.flash_fft:
                spectral = flash_conv(
                    u_proj, phi_proj, self.flash_fft, self.use_tensordot
                )
            else:
                spectral = conv(u_proj, phi_proj, self.n, self.use_tensordot)
        else:
            # Convolve inputs and filters, shape => (B,L,K,dim?) depending on your conv shape
            if self.flash_fft:
                spectral = flash_conv(
                    u, self.filters, self.flash_fft, self.use_tensordot
                )
            else:
                spectral = conv(u, self.filters, self.n, self.use_tensordot)
            # Then contract
            spectral = torch.einsum("blki,dki->bli", spectral, self.M)

        return spectral

    def forward(
        self, u: torch.Tensor, y_ground_truth: torch.Tensor = None
    ) -> torch.Tensor:
        # Teacher-forcing mode
        if y_ground_truth is not None:
            u = y_ground_truth

        # 1) The univariate Chebyshev expansion (first term)
        uni_term = self._univariate_cheb_expansion(u)

        # 2) The bivariate Chebyshev expansion (second term)
        bi_term = self._bivariate_cheb_expansion(u)

        # 3) The spectral convolution component (third term)
        spec_term = self._spectral_component(u)
        out = uni_term + bi_term + spec_term
        return out


class STULayer(nn.Module):
    """
    An STU (Spectral Transform Unit) layer.
    """

    def __init__(self, config, phi, n) -> None:
        super(STULayer, self).__init__()
        self.stu = STU(config, phi, n)
        self.stu_norm = (
            nn.RMSNorm(config.dim, dtype=config.torch_dtype)
            if config.use_norm
            else nn.Identity()
        )
        self.use_mlp = config.use_mlp
        self.dropout = nn.Dropout(p=config.dropout)

        if self.use_mlp:
            self.mlp = MLP(
                dim=config.dim,
                h_dim=config.mlp_scale * config.dim,
                bias=config.bias,
                dtype=config.torch_dtype,
            )
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


class PolynomialSpectralSSMConfig(PretrainedConfig):
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
        use_mlp: bool = True,
        use_tensordot: bool = False,
        use_flash_fft: bool = False,
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

        # STU customization flags
        self.use_tensordot = use_tensordot
        self.use_flash_fft = use_flash_fft
        self.use_norm = use_norm
        self.use_mlp = use_mlp
        self.mlp_scale = mlp_scale
        self.dropout = dropout

        # Device and dtype settings
        self.torch_dtype = torch_dtype
        self.device = device


class PolynomialSpectralSSM(PreTrainedModel):
    config_class = PolynomialSpectralSSMConfig

    def __init__(self, config, phi):
        super(PolynomialSpectralSSM, self).__init__(config)
        self.config = config
        self.d_in = config.d_in
        self.dim = config.dim
        self.d_out = config.d_out
        self.filters = phi  # Precomputed FFT of top K eigenvectors.
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.num_layers = config.num_layers
        self.use_tensordot = config.use_tensordot
        self.init_std_factor = InitStdFactor.CURRENT_DEPTH
        self.init_base_std = 0.02
        self.std = self.init_base_std

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(STULayer(config, self.filters, self.n))

        self.in_proj = nn.Linear(
            config.d_in, config.dim, bias=config.bias, dtype=config.torch_dtype
        )
        self.norm = (
            nn.RMSNorm(config.dim, dtype=config.torch_dtype)
            if config.use_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(
            config.dim, config.d_out, bias=config.bias, dtype=config.torch_dtype
        )

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
                    torch.nn.init.normal_(
                        m.weight, mean=0.0, std=self.init_base_std * factor
                    )
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
    import os

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, "config.json")

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
    configs = PolynomialSpectralSSMConfig(
        num_layers=config_data["num_layers"],
        d_in=config_data["d_in"],
        dim=config_data["dim"],
        d_out=config_data["d_out"],
        seq_len=config_data["seq_len"],
        bias=config_data["bias"],
        num_filters=config_data["num_filters"]
        or math.ceil(math.log(config_data["seq_len"])),
        use_mlp=config_data["use_mlp"],
        use_tensordot=config_data["use_tensordot"],
        use_flash_fft=config_data["use_flash_fft"],
        use_norm=config_data["use_norm"],
        dtype=torch_dtype,
        device=device,
    )

    filters = get_polynomial_spectral_filters(
        seq_len=config_data["seq_len"],
        k=config_data["num_filters"] or math.ceil(math.log(config_data["seq_len"])),
        device=device,
        dtype=torch_dtype,
    )
    print("Configs:")
    for key, value in vars(configs).items():
        print(f"  {key}: {value}")

    model = PolynomialSpectralSSM(configs, filters).to(device=device, dtype=torch_dtype)
    x = torch.randn(
        configs.bsz, configs.seq_len, configs.d_in, device=device, dtype=torch_dtype
    )
    outputs = model(x)

    print("Output shape:", outputs.shape)
    print("Sample output:", outputs[0, 0, :10])
