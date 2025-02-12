import math
import torch
from torch.utils.data import TensorDataset

def generate_lds(
    num_examples: int = 10,
    sequence_len: int = 512,
    num_regimes: int = 1,      # For a fixed LDS, use 1; >1 gives a switching system
    state_size: int = 3,       # hidden state dimension: x_t
    input_size: int = 2,       # control dimension: u_t
    output_size: int = 2,      # observation dimension: y_t
    noise_level: float = 0.1,  # std for process noise w_t
    obs_noise: float = 0.0,    # std for observation noise v_t
    stability_factor: float = 0.95,
    min_duration: int = 100,
    randomness_factor: float = 0.25,
    symmetric: bool = False,   # if True, generate A as symmetric
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generates a dataset from a (possibly switching) linear dynamical system (LDS) in canonical form:
    
        xₜ₊₁ = A₍r₎ xₜ + B₍r₎ uₜ + wₜ,    wₜ ~ N(0, noise_level² I)
        yₜ     = C₍r₎ xₜ + D₍r₎ uₜ + vₜ,    vₜ ~ N(0, obs_noise² I)
    
    When num_regimes == 1, a single set of (A, B, C, D) is generated and used for all
    examples (this is the canonical, fixed LDS case). If num_regimes > 1, then a regime schedule
    is generated so that different time segments use different matrices.
    
    The dataset returned is a TensorDataset of (controls, observations) where:
      - controls: shape (num_examples, sequence_len, input_size)
      - observations: shape (num_examples, sequence_len, output_size)
      
    This implies a “u → y” learning setup, with the hidden state x maintained internally during simulation.
    
    Args:
      num_examples: Number of independent sequences (trajectories).
      sequence_len: Length of each sequence.
      num_regimes:  Number of distinct regimes. For a fixed LDS use 1.
      state_size:   Dimension of hidden state x_t.
      input_size:   Dimension of control input u_t.
      output_size:  Dimension of observation y_t.
      noise_level:  Standard deviation for process noise.
      obs_noise:    Standard deviation for observation noise.
      stability_factor: Factor to scale A to ensure stability (spectral radius <= stability_factor).
      min_duration: Minimum time steps before a regime switch (only used if num_regimes > 1).
      randomness_factor: Controls random offset to regime switch times.
      symmetric:    If True, generates A as symmetric (with eigenvalue spacing).
      seed:         Random seed for reproducibility.
    
    Returns:
      TensorDataset with:
         - inputs:  (num_examples, sequence_len, input_size)
         - targets: (num_examples, sequence_len, output_size)
    """
    torch.manual_seed(seed)
    
    # --- Function to generate one set of matrices (A, B, C, D) ---
    def generate_regime_matrices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if symmetric:
            Q, _ = torch.linalg.qr(torch.randn(state_size, state_size))
            eigenvalues = (torch.rand(state_size) * 2 - 1) * stability_factor  # uniformly in [-stability_factor, stability_factor]
            D = torch.diag(eigenvalues)
            A = Q @ D @ Q.T
        else:
            A = torch.randn(state_size, state_size)
            spec_norm = torch.linalg.norm(A, ord=2)
            A = (A / spec_norm) * stability_factor

        B = torch.randn(state_size, input_size) / math.sqrt(state_size)
        C = torch.randn(output_size, state_size) / math.sqrt(state_size)
        D = torch.randn(output_size, input_size) / math.sqrt(input_size)
        return A, B, C, D

    # Validate constraints
    if num_regimes < 1:
        raise ValueError("num_regimes must be >= 1.")
    if num_regimes > 1 and sequence_len < num_regimes * min_duration:
        raise ValueError("sequence_len too small for the required min durations when switching regimes.")
    if not 0.0 < stability_factor <= 1.0:
        raise ValueError("stability_factor must be in (0,1].")
    if not 0.0 <= randomness_factor <= 1.0:
        raise ValueError("randomness_factor must be in [0,1].")

    if num_regimes == 1:
        # Generate one fixed set of matrices
        A_fixed, B_fixed, C_fixed, D_fixed = generate_regime_matrices()
        # Create one-row tensors for easy indexing later:
        A_mats = A_fixed.unsqueeze(0)  # shape: (1, state_size, state_size)
        B_mats = B_fixed.unsqueeze(0)  # shape: (1, state_size, input_size)
        C_mats = C_fixed.unsqueeze(0)  # shape: (1, output_size, state_size)
        D_mats = D_fixed.unsqueeze(0)  # shape: (1, output_size, input_size)
        # For all examples and all time steps, regime index is 0.
        schedules = torch.zeros(num_examples, sequence_len, dtype=torch.long)
    else:
        # Generate matrices per regime (switching system)
        A_list, B_list, C_list, D_list = [], [], [], []
        for _ in range(num_regimes):
            A_, B_, C_, D_ = generate_regime_matrices()
            A_list.append(A_)
            B_list.append(B_)
            C_list.append(C_)
            D_list.append(D_)
        A_mats = torch.stack(A_list, dim=0)  # (num_regimes, state_size, state_size)
        B_mats = torch.stack(B_list, dim=0)  # (num_regimes, state_size, input_size)
        C_mats = torch.stack(C_list, dim=0)  # (num_regimes, output_size, state_size)
        D_mats = torch.stack(D_list, dim=0)  # (num_regimes, output_size, input_size)
    
        # --- Create regime schedules ---
        def generate_regime_schedule() -> torch.Tensor:
            ideal_segment = sequence_len / num_regimes
            regime_changes = []
            for r in range(1, num_regimes):
                min_valid = r * min_duration
                max_valid = sequence_len - (num_regimes - r) * min_duration
                offset_range = int(randomness_factor * ideal_segment)
                random_offset = torch.randint(-offset_range, offset_range + 1, (1,)).item()
                switch_point = int(r * ideal_segment + random_offset)
                switch_point = max(min_valid, min(switch_point, max_valid))
                regime_changes.append(switch_point)

            schedule = torch.zeros(sequence_len, dtype=torch.long)
            current_regime = 0
            for t in range(sequence_len):
                if regime_changes and t >= regime_changes[0]:
                    current_regime += 1
                    regime_changes.pop(0)
                schedule[t] = current_regime
            return schedule

        schedules = torch.stack([generate_regime_schedule() for _ in range(num_examples)])
        # schedules shape: (num_examples, sequence_len)

    # --- Generate control inputs ---
    u = torch.randn(num_examples, sequence_len, input_size)  # shape: (num_examples, sequence_len, input_size)

    # Initialize hidden state x for each sequence (e.g., zeros)
    x = torch.zeros(num_examples, state_size)

    # Allocate tensors for controls and observations
    controls = torch.zeros(num_examples, sequence_len, input_size)
    observations = torch.zeros(num_examples, sequence_len, output_size)

    # --- Simulate the system ---
    for t in range(sequence_len):
        controls[:, t] = u[:, t]
        regime_idx = schedules[:, t]  # For fixed LDS, this is all 0.
        A_batch = A_mats[regime_idx]   # shape: (num_examples, state_size, state_size)
        B_batch = B_mats[regime_idx]   # shape: (num_examples, state_size, input_size)
        C_batch = C_mats[regime_idx]   # shape: (num_examples, output_size, state_size)
        D_batch = D_mats[regime_idx]   # shape: (num_examples, output_size, input_size)

        # Compute observation: y_t = C x_t + D u_t + observation noise
        y_t = (torch.bmm(C_batch, x.unsqueeze(2)).squeeze(2) +
               torch.bmm(D_batch, u[:, t].unsqueeze(2)).squeeze(2))
        if obs_noise > 0:
            y_t += torch.randn_like(y_t) * obs_noise
        observations[:, t] = y_t

        # Update state: x_{t+1} = A x_t + B u_t + process noise
        x_next = (torch.bmm(A_batch, x.unsqueeze(2)).squeeze(2) +
                  torch.bmm(B_batch, u[:, t].unsqueeze(2)).squeeze(2))
        if noise_level > 0:
            x_next += torch.randn_like(x_next) * noise_level
        x = x_next

    return TensorDataset(controls, observations)
