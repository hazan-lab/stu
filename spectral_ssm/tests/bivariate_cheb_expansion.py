import torch
import torch.nn.functional as F

def double_sum_no_loop(u: torch.Tensor, Q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes:
      out[b,t,:] = sum_{s=0}^{n-1} sum_{i=1}^{n} c[i] * Q[max(0, i-n+s)] @ u[b,t-s,:]
    
    Args:
      u: Tensor of shape (B, L, d_in)
      Q: Tensor of shape (n, d_in, d_out)
      c: Tensor of shape (n,)  (Chebyshev coefficients)
    
    Returns:
      Tensor of shape (B, L, d_out)
    """
    B, L, d_in = u.shape
    n, _, d_out = Q.shape
    assert c.shape[0] == n, "c and Q must have matching 'n' dimension"

    # 1) Build all time-shifted copies of u.
    # For each shift s in [0, n-1], we want u_{t-s}.
    T = torch.arange(L, device=u.device)
    S = torch.arange(n, device=u.device)
    ts_idx = T.unsqueeze(0) - S.unsqueeze(1)      # shape (n, L)
    ts_idx_clamped = ts_idx.clamp(min=0, max=L-1)     # shape (n, L)

    # Gather along time dimension.
    # u: (B, L, d_in) -> gather => (B, n, L, d_in), then permute to (n, B, L, d_in)
    u_shifts = u[:, ts_idx_clamped, :]              # (B, n, L, d_in)
    u_shifts = u_shifts.permute(1, 0, 2, 3)         # (n, B, L, d_in)

    # Zero out positions where original indices were negative.
    mask = (ts_idx < 0).unsqueeze(1).unsqueeze(-1)   # (n, 1, L, 1)
    u_shifts = u_shifts.masked_fill(mask.expand_as(u_shifts), 0)

    # 2) Build offset array for indexing Q:
    # For each i in [1..n] (we use 0-index i: 0..n-1) and s in [0..n-1]:
    #    offset[i,s] = clamp(i - n + s, 0, n-1)
    i_idx = torch.arange(1, n + 1, device=u.device).view(n, 1)  # (n, 1)
    s_idx = torch.arange(n, device=u.device).view(1, n)           # (1, n)
    offset = (i_idx - n + s_idx).clamp(min=0, max=n-1)              # (n, n)
    Q_offset = Q[offset]  # shape (n, n, d_in, d_out)

    # 3) Combine using einsum.
    # Let:
    #   Q_offset: indices "i s r d" with shape (n, n, d_in, d_out)
    #   u_shifts: indices "s b t r" with shape (n, B, L, d_in)
    # We want to contract over the feature dimension "r".
    # So:
    #   Z[i,s,b,t,d] = sum_{r} Q_offset[i,s,r,d] * u_shifts[s,b,t,r]
    Z = torch.einsum("isrd,sbtr->isbtd", Q_offset, u_shifts)  # (n, n, B, L, d_out)

    # Multiply by coefficient c[i] for each i.
    Z = Z * c.view(n, 1, 1, 1, 1)  # broadcasting over s, b, t, d_out.

    # Sum over i and s to get (B, L, d_out)
    out = Z.sum(dim=(0, 1))
    return out

# For comparison: a raw loop version
def double_sum_loop(u: torch.Tensor, Q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Raw loop version computing:
      out[b,t,:] = sum_{s=0}^{n-1} sum_{i=1}^{n} c[i] * Q[max(0, i-n+s)] @ u[b,t-s,:]
    """
    B, L, d_in = u.shape
    n, _, d_out = Q.shape
    out = torch.zeros(B, L, d_out, device=u.device, dtype=u.dtype)
    
    for s in range(n):
        # Shift u: for time t, use u[t-s] if valid, else zero.
        shifted = torch.zeros_like(u)
        if s < L:
            shifted[:, s:] = u[:, :L-s]
        for i in range(1, n+1):
            c_i = c[i-1]
            offset = max(0, i - n + s)
            offset = min(offset, n-1)
            partial = shifted @ Q[offset]  # (B, L, d_out)
            out += c_i * partial
    return out

# Testing both implementations:
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, d_in, d_out = 2, 10, 4, 3
    n = 5  # number of coefficients / degree
    u = torch.randn(B, L, d_in)
    Q = torch.randn(n, d_in, d_out)
    c = torch.randn(n)
    
    out_vec = double_sum_no_loop(u, Q, c)
    out_loop = double_sum_loop(u, Q, c)
    
    diff = (out_vec - out_loop).abs().max().item()
    print("Max absolute difference:", diff)
    print("Vectorized output:\n", out_vec)
    print("Loop output:\n", out_loop)
