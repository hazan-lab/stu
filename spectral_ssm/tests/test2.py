import torch

def compute_ar_brute(M_y: torch.Tensor, u_t: torch.Tensor, M_u: torch.Tensor) -> torch.Tensor:
    """
    Sequential brute force implementation computing:
    ŷ_t = sum(M_i^y * ŷ_{t-i}) + sum(M_i^u * u_{t+1-i})
    where ŷ_{t-i} is the previously computed output
    
    Args:
        M_y: AR weights (d_out, k_y, d_in)
        u_t: Control inputs (bsz, seq_len, d_u)
        M_u: Control weights (k_u, d_out, d_u)
    """
    bsz, seq_len, d_u = u_t.shape
    d_out, k_y, d_in = M_y.shape
    k_u = M_u.shape[0]
    
    # Store computed outputs
    y_pred = torch.zeros(bsz, seq_len, d_out)
    
    print(f"\nComputing step-by-step predictions:")
    print(f"Sequence length: {seq_len}")
    print(f"AR lags (k_y): {k_y}")
    print(f"Control lags (k_u): {k_u}")
    
    for t in range(seq_len):
        print(f"\nTime step t={t}")
        
        # AR component using previously computed outputs
        ar_term = torch.zeros(bsz, d_out)
        print(f"  AR component:")
        for i in range(1, k_y + 1):
            if t - i >= 0:
                print(f"    Lag {i}:")
                print(f"      Previous output ŷ_{t-i} = {y_pred[0, t-i].tolist()}")
                print(f"      Weight M_{i}^y = {M_y[:, i-1].tolist()}")
                term = y_pred[:, t-i] @ M_y[:, i-1].T
                ar_term += term
                print(f"      Adding term: {term[0].tolist()}")
                print(f"      Running AR sum: {ar_term[0].tolist()}")
        
        # Control component
        u_term = torch.zeros(bsz, d_out)
        print(f"  Control component:")
        for i in range(1, k_u + 1):
            if t + 1 - i >= 0 and t + 1 - i < seq_len:
                print(f"    Lag {i}:")
                print(f"      Control input u_{t+1-i} = {u_t[0, t+1-i].tolist()}")
                print(f"      Weight M_{i}^u = {M_u[i-1].tolist()}")
                term = u_t[:, t+1-i] @ M_u[i-1].T
                u_term += term
                print(f"      Adding term: {term[0].tolist()}")
                print(f"      Running control sum: {u_term[0].tolist()}")
        
        # Combine terms
        y_pred[:, t] = ar_term + u_term
        print(f"  Final output ŷ_{t} = {y_pred[0, t].tolist()}")
    
    return y_pred

def test_sequential():
    """Simple test case"""
    # Setup parameters
    d_out, k_y, d_in = 2, 2, 2
    d_u, k_u = 2, 3
    seq_len = 4
    
    # AR weights
    M_y = torch.tensor([
        [[1.0, 0.5],    # M_1 for output dim 0
         [0.2, 0.1]],   # M_2 for output dim 0
        [[1.0, 0.5],    # M_1 for output dim 1
         [0.2, 0.1]]    # M_2 for output dim 1
    ]).reshape(d_out, k_y, d_in)
    
    # Control weights
    M_u = torch.tensor([
        [[0.5, 0.3],    # M_1^u 
         [0.5, 0.3]],
        [[0.2, 0.1],    # M_2^u
         [0.2, 0.1]],
        [[0.1, 0.1],    # M_3^u
         [0.1, 0.1]]
    ]).reshape(k_u, d_out, d_u)
    
    # Control inputs
    u_t = torch.ones(1, seq_len, d_u)  # All ones for simplicity
    
    # Compute predictions
    y_pred = compute_ar_brute(M_y, u_t, M_u)

if __name__ == "__main__":
    test_sequential()