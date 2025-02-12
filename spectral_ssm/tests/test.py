import torch
from utils import compute_ar_u

# ===============================
# Core Functionality
# ===============================

def get_toeplitz(x: torch.Tensor, k: int, lower: bool = True) -> torch.Tensor:
    """
    Efficiently construct Toeplitz matrices for each batch and feature.
    For AR-y, we use lower=True to only look at past values.

    Args:
        x: Input tensor of shape (bsz, sl, d)
        k_y: Number of steps to include in the Toeplitz construction
        lower: If True, construct lower triangular Toeplitz (past values)
               If False, construct upper triangular Toeplitz (future values)

    Returns:
        torch.Tensor: Toeplitz matrices of shape (bsz, sl, k_y, d)
    """
    bsz, sl, d = x.shape

    print(f"\nConstructing Toeplitz matrix with parameters:")
    print(f"  Batch size (bsz): {bsz}")
    print(f"  Sequence length (sl): {sl}")
    print(f"  Feature dimension (d): {d}")
    print(f"  Number of steps (k): {k}")
    print(f"  Lower triangular (lower): {lower}")

    # Create row and column indices for constructing the Toeplitz matrix
    row_indices = torch.arange(sl, device=x.device)  # [0, 1, 2, ..., sl-1]
    col_indices = torch.arange(k, device=x.device)   # [0, 1, 2, ..., k-1]

    # Compute relative positions between each output position and its inputs
    # Shape: (sl, k)
    indices = col_indices - row_indices.unsqueeze(1)

    # Create causality mask
    # For lower triangular (past values), we want indices <= 0
    # For upper triangular (future values), we want indices >= 0
    # Shape: (1, sl, k_y, 1)
    if lower:
        mask = indices.le(0).unsqueeze(0).unsqueeze(-1)
    else:
        mask = indices.ge(0).unsqueeze(0).unsqueeze(-1)

    # Expand input to match desired output shape
    # Shape: (bsz, sl, k_y, d)
    x_expanded = x.unsqueeze(2).expand(bsz, sl, k, d)

    # Gather values according to the computed indices
    # -indices gives us the correct offsets for gathering past values
    # clamp(min=0) ensures we don't access negative indices
    # Shape: (bsz, sl, k_y, d)
    shifted = x_expanded.gather(
        1,
        (-indices).clamp(min=0).unsqueeze(0).unsqueeze(-1).expand(bsz, sl, k, d)
    )

    # Apply mask to zero out invalid positions
    # Shape: (bsz, sl, k_y, d)
    result = shifted * mask.to(x.dtype)

    print(f"Toeplitz matrix constructed with shape: {result.shape}")
    return result


def compute_ar_y_toeplitz(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Matrix-based implementation using Toeplitz structure for the autoregressive equation:
    ŷ_t = sum(M_i^y * ŷ_{t-i}) for i=1 to k_y

    Args:
        M_y: Transition weights of shape (d_out, k_y, d_in)
        y_t: Initial values of shape (bsz, sl, d_in)
    """
    d_out, k_y, d_in = M_y.shape
    bsz, sl, _ = y_t.shape
    device = y_t.device

    # Initialize output tensor
    output = torch.zeros(bsz, sl, d_out, device=device)
    
    # Process sequence using previous outputs
    for t in range(sl):
        # Create Toeplitz for current timestep using previous computed outputs
        if t > 0:
            # Use only previously computed outputs up to t-1
            prev_outputs = output[:, :t]
            y_toeplitz = get_toeplitz(prev_outputs, min(k_y, t), lower=True)
            # Compute AR term using prev outputs
            y_toeplitz_expanded = y_toeplitz[:, -1:].unsqueeze(-2)  # Only need last timestep
            M_y_expanded = M_y.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
            ar_term = (y_toeplitz_expanded @ M_y_expanded.transpose(-1, -2)).sum(dim=2).squeeze(-2)
            output[:, t] = ar_term.squeeze(1)

    return output

def compute_ar_y(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Sequential implementation for autoregressive component that uses
    previously computed outputs.

    Args:
        M_y: Transition weights of shape (d_out, k_y, d_in)
        y_t: Initial values of shape (bsz, sl, d_in)
    """
    d_out, k_y, d_in = M_y.shape
    bsz, sl, _ = y_t.shape
    device = y_t.device
    
    # Initialize output
    output = torch.zeros(bsz, sl, d_out, device=device)
    
    # Process sequence
    for t in range(sl):
        # AR component using previously computed outputs
        for i in range(1, k_y + 1):
            if t - i >= 0:
                M_i = M_y[:, i-1]
                y_prev = output[:, t-i]  # Use previously computed output
                term = y_prev @ M_i.T
                output[:, t] += term
    
    return output

def compute_ar_y_brute(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Brute force implementation that computes ŷ_t = sum(M_i^y * y_{t-i}) for i=1 to k_y

    Args:
        M_y: Transition weight matrices of shape (d_out, k_y, d_out)
        y_t: Values at each time step of shape (bsz, seq_len, d_out)

    Returns:
        torch.Tensor: Autoregressive outputs of shape (bsz, seq_len, d_out)
    """
    bsz, seq_len, d_out = y_t.shape
    _, k_y, _ = M_y.shape
    output = torch.zeros_like(y_t)
    
    print("\n" + "-"*50)
    print("Computing autoregressive predictions (Brute Force):")
    print("-"*50)
    print(f"Input shape: batch_size={bsz}, sequence_length={seq_len}, d_out={d_out}")
    print(f"Number of lags (k_y): {k_y}")
    
    for t in range(seq_len):
        print(f"\nTime step t={t}:")
        for i in range(1, k_y + 1):
            if t - i >= 0:
                M_i = M_y[:, i-1]
                y_prev = output[:, t-i]
                
                # Round tensors for better readability
                M_i_rounded = torch.round(M_i * 100) / 100
                y_prev_rounded = torch.round(y_prev * 100) / 100
                
                print(f"  Lag i={i}:")
                print(f"    Using M_{i}^y:\n{M_i_rounded}")
                print(f"    Multiplying with y_{{{t-i}}}:\n{y_prev_rounded}")
                
                term = y_prev @ M_i.T
                term_rounded = torch.round(term * 100) / 100
                print(f"    Term M_{i}^y @ y_{{{t-i}}} =\n{term_rounded}")
                
                output[:, t] += term
                output_rounded = torch.round(output[:, t] * 100) / 100
                print(f"    Running sum ŷ_{t} =\n{output_rounded}")
            else:
                print(f"  Lag i={i}: Skipped (insufficient history at t={t})")
    
    return output

# ===============================
# Testing Suite
# ===============================

def test_toeplitz_construction():
    """
    Test the Toeplitz matrix construction with a simple sequence.
    """
    print("\n" + "="*50)
    print("TESTING TOEPLITZ CONSTRUCTION")
    print("="*50)
    
    # Create a simple sequence
    x = torch.tensor([[[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]]])  # Shape: (1, 4, 2)
    k = 3
    
    print("\nInput sequence:")
    for t in range(x.shape[1]):
        x_rounded = torch.round(x[0, t] * 100) / 100
        print(f"x_{t} = {x_rounded.tolist()}")
    
    # Get Toeplitz matrices
    toep = get_toeplitz(x, k, lower=True)
    
    print(f"\nToeplitz matrices (k={k}):")
    for t in range(toep.shape[1]):
        print(f"\nPosition t={t}:")
        for lag in range(k):
            toep_val = torch.round(toep[0, t, lag] * 100) / 100
            print(f"  Lag {lag}: {toep_val.tolist()}")
            
    # Verify specific properties
    print("\nVerifying Toeplitz properties:")
    
    # 1. Check causality (only past values)
    valid_entries = toep[0, 1, 0]  # Should be non-zero
    invalid_entries = toep[0, 0, 1:]  # Should be zero
    valid_entries_rounded = torch.round(valid_entries * 100) / 100
    invalid_entries_rounded = torch.round(invalid_entries * 100) / 100
    print(f"1. Causality check:")
    print(f"   First position, lag>0 (should be zeros): {invalid_entries_rounded.tolist()}")
    print(f"   Second position, lag=0 (should be non-zero): {valid_entries_rounded.tolist()}")
    
    # 2. Check correct time offsets
    print(f"\n2. Time offset check (t=2):")
    for lag in range(k):
        if lag < 2:
            source_t = 2 - lag
            toep_val = torch.round(toep[0, 2, lag] * 100) / 100
            print(f"   Lag {lag}: {toep_val.tolist()} (from t={source_t})")
        else:
            toep_val = torch.round(toep[0, 2, lag] * 100) / 100
            print(f"   Lag {lag}: {toep_val.tolist()} (no corresponding t)")


def test_both_implementations():
   """
   Test and compare implementations with additional control component
   """
   print("\n" + "="*50)
   print("TESTING IMPLEMENTATIONS WITH CONTROL")
   print("="*50)
   
   # Setup parameters
   d_out, k_y = 2, 2  
   d_u, k_u = 2, 3  # Control input dimensions
   
   # Test Case 1: Same weights for both dimensions
   M_y_1 = torch.tensor([
       [[1.0, 0.5],    # M_1: weight for y_{t-1}
        [0.2, 0.1]],   # M_2: weight for y_{t-2}
       [[1.0, 0.5],    # Same weights for second output dimension
        [0.2, 0.1]]
   ]).reshape(d_out, k_y, d_out)
   
   # Control matrices - simple case (d_out, k_u, d_u)
   M_u = torch.tensor([
       # First output dimension (d_out=0)
       [[0.5, 0.3],    # k_u=0
        [0.2, 0.1],    # k_u=1  
        [0.1, 0.1]],   # k_u=2
       # Second output dimension (d_out=1)  
       [[0.5, 0.3],    # k_u=0
        [0.2, 0.1],    # k_u=1
        [0.1, 0.1]]    # k_u=2
   ])  # Already in shape (d_out, k_u, d_u)
   
   y_t_1 = torch.tensor([
       [[1.0, 2.0],    # t=0
        [3.0, 4.0],    # t=1
        [5.0, 6.0],    # t=2
        [7.0, 8.0]]    # t=3
   ])
   
   # Control input sequence
   u_t = torch.tensor([
       [[1.0, 1.0],    # t=0
        [1.0, 1.0],    # t=1
        [1.0, 1.0],    # t=2
        [1.0, 1.0]]    # t=3
   ])

   def shift(x: torch.Tensor, offset: int) -> torch.Tensor:
       """Helper to shift sequence by offset"""
       if offset == 0:
           return x
       pad = torch.zeros_like(x[:, :offset])
       return torch.cat([x[:, offset:], pad], dim=1)

   def compute_combined_brute(M_y, M_u, y_t, u_t):
        """Brute force with control input"""
        bsz, seq_len, d_out = y_t.shape
        output = torch.zeros_like(y_t)
        ar_u = compute_ar_u(M_u, u_t)

        print("\nComputing combined predictions (Brute Force):")
        for t in range(seq_len):
            print(f"\nTime step t={t}:")
            
            # AR component using previously computed outputs
            for i in range(1, k_y + 1):
                if t - i >= 0:
                    M_i = M_y[:, i-1]
                    y_prev = output[:, t-i]
                    term = y_prev @ M_i.T  
                    output[:, t] += term

            # Control component (M_u is d_out, k_u, d_u)
            for i in range(1, k_u + 1):
                if t + 1 - i >= 0 and t + 1 - i < seq_len:
                    u_prev = u_t[:, t+1-i]  # (bsz, d_u)
                    term = u_prev @ M_u[:, i-1].T  # (bsz, d_out)
                    output[:, t] += term

        return output

   def compute_combined_parallel(M_y, M_u, y_t, u_t):
       """Parallel implementation with control"""
       # AR component
       ar_y = compute_ar_y(M_y, y_t)
       
       # Control component
       k_u = M_u.shape[1]  # k_u dimension is now at idx 1
       u_shifted = torch.stack([shift(u_t, i-1) for i in range(1, k_u+1)], dim=1)
       # Reshape M_u for einsum if needed
       ar_u = torch.einsum("bksd,omd->bso", u_shifted, M_u)  # o=d_out, m=k_u, d=d_u
       
       return ar_y + ar_u

   print("\nComputing with both methods...")
   result_brute = compute_combined_brute(M_y_1, M_u, y_t_1, u_t)
   result_parallel = compute_combined_parallel(M_y_1, M_u, y_t_1, u_t)
   
   print("\nResults Comparison:")
   print("Time | Brute Force  | Parallel | Match?")
   print("-"*50)
   for t in range(y_t_1.shape[1]):
       brute_t = [f"{x:.2f}" for x in result_brute[0, t].tolist()]
       parallel_t = [f"{x:.2f}" for x in result_parallel[0, t].tolist()]
       matches = torch.allclose(result_brute[0, t], result_parallel[0, t], rtol=1e-5)
       print(f"t={t}  | {brute_t} | {parallel_t} | {matches}")

# ===============================
# Main Execution
# ===============================

def main():
    """
    Main function to execute tests. Comment out the function calls to disable specific tests.
    """
    test_toeplitz_construction()
    test_both_implementations()


if __name__ == "__main__":
    main()
