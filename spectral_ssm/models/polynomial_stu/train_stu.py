import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_ssm.lds import generate_lds
from spectral_ssm.models.polynomial_stu import (
    PolynomialSpectralSSM,
    PolynomialSpectralSSMConfig,
    get_polynomial_spectral_filters,
)

torch.set_float32_matmul_precision("high")

curr_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(curr_dir, "config.json")

with open(config_path, "r") as f:
    config_data = json.load(f)

def normalize(dataset: TensorDataset) -> tuple[TensorDataset, dict[str, torch.Tensor]]:
    """Normalizes a dataset using pure torch functions over the batch and time dimensions."""
    inputs, targets = dataset[:]

    input_mean = inputs.mean(dim=(0, 1), keepdim=True)
    input_std = inputs.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    inputs_norm = (inputs - input_mean) / input_std

    target_mean = targets.mean(dim=(0, 1), keepdim=True)
    target_std = targets.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    targets_norm = (targets - target_mean) / target_std

    print("\nInput Statistics per Dimension:")
    print("Mean:", input_mean.squeeze())
    print("Std:", input_std.squeeze())
    print("\nTarget Statistics per Dimension:")
    print("Mean:", target_mean.squeeze())
    print("Std:", target_std.squeeze())
    print()

    stats = {
        "input_mean": input_mean.squeeze(),
        "input_std": input_std.squeeze(),
        "target_mean": target_mean.squeeze(),
        "target_std": target_std.squeeze(),
        "input_dims": inputs.shape[-1],
        "output_dims": targets.shape[-1],
    }

    return TensorDataset(inputs_norm, targets_norm), stats


# Set default values for essential parameters
config_data.setdefault("epochs", 1)
config_data.setdefault("bsz", 1)
config_data.setdefault("num_layers", 2)
config_data.setdefault("d_in", 4)
config_data.setdefault("dim", 32)
config_data.setdefault("d_h", 3)
config_data.setdefault("d_out", 4)
config_data.setdefault("seq_len", 256)
config_data.setdefault("bias", False)
config_data.setdefault("num_filters", None)
config_data.setdefault("use_mlp", True)
config_data.setdefault("use_norm", False)
config_data.setdefault("normalize_dataset", False)
config_data.setdefault("mlp_scale", 4)
config_data.setdefault("dropout", 0.0)
config_data.setdefault("torch_dtype", "bfloat16")
config_data.setdefault("seed", 1746)

SEED = config_data["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

# Extract configuration parameters
epochs = config_data["epochs"]
bsz = config_data["bsz"]
num_layers = config_data["num_layers"]
d_in = config_data["d_in"]
dim = config_data["dim"]
d_h = config_data["d_h"]
d_out = config_data["d_out"]
seq_len = config_data["seq_len"]
bias = config_data["bias"]
num_filters = config_data["num_filters"] or math.ceil(math.log(seq_len))
use_mlp = config_data["use_mlp"]
use_norm = config_data["use_norm"]
mlp_scale = config_data["mlp_scale"]
dropout = config_data["dropout"]

if config_data["torch_dtype"] == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_examples = config_data["num_examples"]
num_regimes = config_data["num_regimes"]
noise_level = config_data["noise_level"]
obs_noise = config_data["obs_noise"]
stability_factor = config_data["stability_factor"]
min_duration = seq_len // num_regimes
randomness_factor = 0.0 if num_regimes == 1 else config_data["randomness_factor"]
symmetric = config_data["symmetric"]
lr = config_data["lr"]

# Create model configuration
config = PolynomialSpectralSSMConfig(
    bsz=bsz,
    num_layers=num_layers,
    d_in=d_in,
    dim=dim,
    d_out=d_out,
    seq_len=seq_len,
    bias=bias,
    num_filters=num_filters,
    use_mlp=use_mlp,
    use_norm=use_norm,
    mlp_scale=mlp_scale,
    dropout=dropout,
    torch_dtype=torch_dtype,
    device=device,
)

# Get polynomial spectral filters
spectral_filters = get_polynomial_spectral_filters(
    seq_len=seq_len,
    k=num_filters,
    device=device,
    dtype=torch_dtype,
)

print("Spectral filters shape (phi):", spectral_filters[0].shape)
print("Spectral filters shape (sigma):", spectral_filters[1].shape)

print("Configs:")
for key, value in config_data.items():
    print(f"  {key}: {value}")

# Initialize model and optimizer
model = PolynomialSpectralSSM(config, spectral_filters).to(device=device, dtype=torch_dtype)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Generate dataset
dataset = generate_lds(
    num_examples=num_examples,
    sequence_len=seq_len,
    num_regimes=num_regimes,
    input_size=d_in,
    state_size=d_h,
    output_size=d_out,
    noise_level=noise_level,
    obs_noise=obs_noise,
    stability_factor=stability_factor,
    min_duration=min_duration,
    randomness_factor=randomness_factor,
    symmetric=symmetric,
    seed=SEED,
)

if config_data["normalize_dataset"]:
    dataset, norm_stats = normalize(dataset)

train_loader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True)

model.train()
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, dtype=torch_dtype), y.to(device, dtype=torch_dtype)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == len(train_loader) - 1:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.4f}"
            )

# Generate and save visualization
final_x, final_y = next(iter(train_loader))
final_x, final_y = (
    final_x.to(device=device, dtype=torch_dtype),
    final_y.to(device=device, dtype=torch_dtype),
)
with torch.no_grad():
    final_out = model(final_x)

final_prediction = final_out[0, :, 0].to(dtype=torch.float32).cpu().numpy()
final_truth = final_y[0, :, 0].to(dtype=torch.float32).cpu().numpy()

plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

ax.plot(final_truth, color="#10B981", label="Ground Truth", linewidth=2)
ax.plot(final_prediction, color="#8B5CF6", label="Prediction", linewidth=3)

mse = np.mean((final_prediction - final_truth) ** 2)

# Add shaded area between curves
ax.fill_between(
    range(len(final_truth)), final_truth, final_prediction, color="#D1C4E9", alpha=0.9
)

ax.set_title(
    f"Polynomial STU Predictions vs Ground Truth\nMSE: {mse:.4f}", fontsize=16, pad=15
)
ax.set_xlabel("Time Step", fontsize=14)
ax.set_ylabel("Value", fontsize=14)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=12)
ax.grid(True, alpha=0.15)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

plt.tight_layout(pad=2.0)
plt.savefig("plots/polynomial_stu_performance.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFinal MSE: {mse:.4f}\n")
print("Plot saved to plots/polynomial_stu_performance.png")
