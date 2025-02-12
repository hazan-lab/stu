import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(
        f"Unable to import Triton-based flash attention: {e}. No alternative currently available."
    )


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):    
    # For half the dimensions, build the scale factor:
    freq_seq = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (theta ** freq_seq)

    # Outer product with positions
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    
    # Build a complex exponential e^{i * theta}
    freqs_cis = torch.polar(
        torch.ones_like(angles),
        angles
    )
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    x is [B, n_heads, seq_len, head_dim_as_complex],
    so we want to broadcast freqs_cis from [max_seq_len, half_dim]
    to [1, 1, seq_len, half_dim].
    """
    seq_len = x.shape[2]
    freqs_cis = freqs_cis[:seq_len]  # slice down to current seq_len
    return freqs_cis.view(1, 1, seq_len, -1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert real -> complex by grouping last dim in pairs
    # shape => [B, n_heads, seq_len, head_dim//2, 2] => complex => [B, n_heads, seq_len, head_dim//2]
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast the frequencies to match [B, n_heads, seq_len, head_dim//2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    # Multiply => apply rotation
    xq_complex = xq_complex * freqs_cis
    xk_complex = xk_complex * freqs_cis

    # Convert back to real => shape [B, n_heads, seq_len, head_dim]
    xq_out = torch.view_as_real(xq_complex).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_complex).reshape(*xk.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.c_attn = nn.Linear(self.dim, 3*self.dim, bias=config.bias)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.c_proj.SCALE_INIT = 1

        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if config.use_alibi else None
        self.window_size = config.window_size
        self.softcap = config.softcap

        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

    def _generate_slopes(self, n: int):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, num_heads: int, interpolation_factor: float = 0.25):
        # If n_heads is a power of 2, generate slopes directly
        if math.log2(num_heads).is_integer():
            slopes = self._generate_slopes(num_heads)
        else:
            # Get slopes for the nearest power of two
            n = nearest_power_of_two(num_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)

            # Generate extra slopes
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: num_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=torch.device("cuda"))
        slopes = slopes * interpolation_factor  # https://arxiv.org/pdf/2310.13017
        return slopes

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
    ) -> torch.Tensor:
        if x is not None:
            q = k = v = x
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")

        bsz, q_len, dim = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape

        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim)
        v = v.view(bsz, v_len, self.num_heads, self.head_dim)

        if self.alibi_slopes is None: # Use either ALiBi or RoPE
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        y = flash_attn_func(  # https://arxiv.org/pdf/2307.08691
            q=q, k=k, v=v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0), # Set to config.seq_len if full attention
            alibi_slopes=self.alibi_slopes,    # https://arxiv.org/pdf/2108.12409
            softcap=self.softcap,              # https://arxiv.org/pdf/2408.00118
        )

        y = y.contiguous().view(bsz, q_len, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y


class AttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super(AttentionLayer, self).__init__()
        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = Attention(config=config)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor=None) -> torch.Tensor:
        x = x + self.dropout(self.attn(x=self.attn_norm(x), freqs_cis=freqs_cis))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        # https://arxiv.org/pdf/2002.05202
        super().__init__()
        self.hidden_size = config.dim
        self.intermediate_size = config.dim * config.mlp_scale
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias, dtype=config.torch_dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias, dtype=config.torch_dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias, dtype=config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        outputs = self.dropout(outputs)
        return outputs


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        bsz: int = 1,
        dim: int = 896,
        d_in: int = 4,  # input dimension
        d_out: int = 4,  # output dimension
        num_heads: int = 8,
        num_layers: int = 12,
        seq_len: int = 8192,
        window_size: int = 8192,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        softcap: float = 50.0,
        theta: float = 10_000.0,
        use_alibi: bool = False, # Default to RoPE
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.dim = dim
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.window_size = window_size
        self.hidden_size = dim
        self.mlp_scale = mlp_scale
        self.intermediate_size = self.dim * self.mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.softcap = softcap
        self.theta = theta
        self.use_alibi = use_alibi
        self.torch_dtype = torch_dtype
        self.device = device


class Transformer(PreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config) -> None:
        super(Transformer, self).__init__(config)
        self.num_layers = config.num_layers
        assert config.dim % config.num_heads == 0, f"dim ({config.dim}) must be divisible by num_heads ({config.num_heads})"
        self.head_dim = config.dim // config.num_heads

        # RoPE position embeddings
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            head_dim=self.head_dim,
            max_seq_len=config.seq_len,
            theta=config.theta,
        ), persistent=True)

        self.input_proj = nn.Linear(config.d_in, config.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = AttentionLayer(config=config)
            self.layers.append(layer)

        self.norm = nn.RMSNorm(config.dim)
        self.output_proj = nn.Linear(config.dim, config.d_out, bias=config.bias)

        self.std = config.dim ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, self.freqs_cis)

        x = self.norm(x)
        y_hat = self.output_proj(x)

        return y_hat

    def _get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)

if __name__ == "__main__":
    config_path = "transformer/config.json"

    with open(config_path, "r") as f:
        config_data = json.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    torch_dtype = getattr(torch, config_data["torch_dtype"])
    print("Torch dtype:", torch_dtype)

    configs = TransformerConfig(
        bsz=config_data["bsz"],
        dim=config_data["dim"],
        num_heads=config_data["num_heads"],
        num_layers=config_data["num_layers"],
        seq_len=config_data["seq_len"],
        window_size=config_data["window_size"],
        mlp_scale=config_data["mlp_scale"],
        bias=config_data["bias"],
        dropout=config_data["dropout"],
        softcap=config_data["softcap"],
        theta=config_data["theta"],
        use_alibi=config_data["use_alibi"],
        torch_dtype=torch_dtype,
    )

    print("Configs:")
    for key, value in vars(configs).items():
        print(f"  {key}: {value}")

    model = Transformer(configs).to(device=device, dtype=torch_dtype)
    
    x = torch.randint(
        0, 32, 
        (config_data["bsz"], config_data["seq_len"]), 
        dtype=torch.long
    ).to(device)

    outputs = model(x)

    print("Output shape:", outputs.shape)
    print("Sample output:", outputs[0, 0, :10])
