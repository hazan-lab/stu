import torch

def naive_ar(y: torch.Tensor, M_y: torch.Tensor) -> torch.Tensor:
    """
    For each time t, we compute:
        out[t] = sum_{j=1..k_y} [ y[t-j] @ M_y[:, j-1, :].T ]
    skipping any j where (t-j) < 0.

    Shapes:
        y:   (bsz, seq_len, dim)
        M_y: (dim, k_y, dim)  # i.e. M_y[out_dim, lag, in_dim]
    """
    bsz, seq_len, dim = y.shape
    # M_y is shaped (dim, k_y, dim), so "d_out, k_y, d_in"
    d_out, k_y, d_in = M_y.shape
    assert d_out == d_in == dim, "Example assumes M_y is (dim, k_y, dim)."

    out = torch.zeros_like(y)  # (bsz, seq_len, dim)

    for t in range(seq_len):
        # We'll accumulate (bsz, dim) for out[:, t]
        accum = torch.zeros((bsz, dim), device=y.device, dtype=y.dtype)
        for j in range(1, k_y + 1):
            t_in = t - j
            if 0 <= t_in < seq_len:
                # y[:, t_in, :]   => (bsz, dim)
                # M_y[:, j-1, :] => (dim, dim)
                accum += y[:, t_in, :] @ M_y[:, j - 1, :].T
        out[:, t, :] = accum
    return out


def vectorized_ar(y: torch.Tensor, M_y: torch.Tensor) -> torch.Tensor:
    """
    Same teacher-forcing AR, but done in a single pass using:
      y_out_ar = einsum("b j i, d j i -> b d")
    at each step, where:
       - 'b j i' is y_buffer  (batch, k_y, in_dim)
       - 'd j i' is M_y       (out_dim, k_y, in_dim)
    and we sum over j (time lag) and i (in_dim).

    Then we 'roll' the buffer by 1, and we insert y[:, t, :] at position 0.
    """
    bsz, seq_len, dim = y.shape
    d_out, k_y, d_in = M_y.shape
    assert d_out == d_in == dim, "Example assumes M_y is (dim, k_y, dim)."

    out = torch.zeros_like(y)   # (bsz, seq_len, dim)

    # y_buffer will hold the most recent k_y ground-truth y-values
    # in descending order of lag:
    #   y_buffer[:, 0, :] => y_{t-1}
    #   y_buffer[:, 1, :] => y_{t-2}, etc.
    y_buffer = torch.zeros((bsz, k_y, dim), device=y.device, dtype=y.dtype)

    for t in range(seq_len):
        # (bsz, dim), summing over k_y and dim in the einsum
        y_out_ar = torch.einsum("bji,dji->bd", y_buffer, M_y)
        out[:, t, :] = y_out_ar

        # Now shift the buffer so that y_buf[:,0,:] => y_{t}
        # for the *next* iteration. The old y_buf[:,0,:] goes to y_buf[:,1,:], etc.
        y_buffer = torch.roll(y_buffer, shifts=1, dims=1)

        # Insert the ground-truth y[:, t, :]
        y_buffer[:, 0, :] = y[:, t, :]

    return out


def main():
    torch.manual_seed(42)

    bsz = 2
    seq_len = 4
    dim = 3
    k_y = 2

    # random y
    y = torch.randn(bsz, seq_len, dim)
    # random M_y with shape (dim, k_y, dim)
    M_y = torch.randn(dim, k_y, dim)

    naive = naive_ar(y, M_y)
    vec   = vectorized_ar(y, M_y)

    print("naive:\n", naive)
    print("vectorized:\n", vec)
    diff = naive - vec
    print("difference (naive - vectorized):\n", diff)

if __name__ == "__main__":
    main()
