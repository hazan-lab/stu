import time
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_ssm.models.transformer import Transformer, TransformerConfig
from spectral_ssm.lds import generate_lds

torch.set_float32_matmul_precision("high")

config_path = "models/transformer/config.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

def apply_compile(model: nn.Module) -> None:
    print(f"Compiling each {model.__class__.__name__} layer with torch.compile...")
    start = time.perf_counter()
    for idx, layer in model.layers.named_children():
        compiled_layer = torch.compile(layer, mode="max-autotune", fullgraph=True)
        model.layers.register_module(idx, compiled_layer)
    print(f"Finished compiling each {model.__class__.__name__} layer in {time.perf_counter() - start:.4f} seconds.")

def normalize(dataset: TensorDataset) -> tuple[TensorDataset, dict[str, torch.Tensor]]:
    inputs, targets = dataset[:]
    input_mean = inputs.mean(dim=(0, 1), keepdim=True)
    input_std  = inputs.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    inputs_norm = (inputs - input_mean) / input_std

    target_mean = targets.mean(dim=(0, 1), keepdim=True)
    target_std  = targets.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
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

# Set defaults in config_data
config_data.setdefault("epochs", 1)
config_data.setdefault("bsz", 1)
config_data.setdefault("d_in", 4)
config_data.setdefault("dim", 32)
config_data.setdefault("head_dim", 8)
config_data.setdefault("num_layers", 2)
dim = config_data["dim"]
head_dim = config_data["head_dim"]
config_data.setdefault("num_heads", dim // head_dim)
config_data.setdefault("d_h", 3)
config_data.setdefault("d_out", 4)
config_data.setdefault("seq_len", 256)
config_data.setdefault("mlp_scale", 4)
config_data.setdefault("bias", False)
config_data.setdefault("dropout", 0.0)
config_data.setdefault("softcap", 50.0)
config_data.setdefault("use_alibi", False)
config_data.setdefault("torch_dtype", "bfloat16")
config_data.setdefault("torch_compile", False)
config_data.setdefault("seed", 1746)
config_data.setdefault("normalize_dataset", False)
config_data.setdefault("num_examples", 5000)
config_data.setdefault("num_regimes", 1)
config_data.setdefault("noise_level", 0.05)
config_data.setdefault("obs_noise", 0.05)
config_data.setdefault("stability_factor", 0.9999)
config_data.setdefault("min_duration", config_data["seq_len"] // config_data["num_regimes"])
config_data.setdefault("randomness_factor", 0.0 if config_data["num_regimes"] == 1 else 0.1)
config_data.setdefault("symmetric", True)
config_data.setdefault("lr", 3e-4)

SEED = config_data["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

epochs           = config_data["epochs"]
bsz              = config_data["bsz"]
d_in             = config_data["d_in"]
dim              = config_data["dim"]
head_dim         = config_data["head_dim"]
num_layers       = config_data["num_layers"]
num_heads        = config_data["num_heads"]
d_h              = config_data["d_h"]
d_out            = config_data["d_out"]
seq_len          = config_data["seq_len"]
mlp_scale        = config_data["mlp_scale"]
bias             = config_data["bias"]
dropout          = config_data["dropout"]
softcap          = config_data["softcap"]
use_alibi        = config_data["use_alibi"]

torch_dtype = torch.bfloat16 if config_data["torch_dtype"] == "bfloat16" else torch.float32
torch_compile = config_data["torch_compile"]
normalize_dataset = config_data["normalize_dataset"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_examples = config_data["num_examples"]
num_regimes  = config_data["num_regimes"]
noise_level  = config_data["noise_level"]
obs_noise    = config_data["obs_noise"]
stability_factor = config_data["stability_factor"]
min_duration = config_data["min_duration"]
randomness_factor = config_data["randomness_factor"]
symmetric = config_data["symmetric"]
lr = config_data["lr"]

config = TransformerConfig(
    bsz=bsz,
    dim=dim,
    num_heads=num_heads,
    num_layers=num_layers,
    seq_len=seq_len,
    window_size=seq_len,  # full attention
    mlp_scale=mlp_scale,
    bias=bias,
    dropout=dropout,
    softcap=softcap,
    use_alibi=use_alibi,
    torch_dtype=torch_dtype,
    device=device,
)

model = Transformer(config).to(device=device, dtype=torch_dtype)
if torch_compile:
    apply_compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

if normalize_dataset:
    dataset, norm_stats = normalize(dataset)

# Optionally: save normalization stats for reproducibility  / later use
# if torch.cuda.is_available():
#     norm_stats = {k: v.cpu() for k, v in norm_stats.items()}
# torch.save(norm_stats, "normalization_stats.pt")

train_loader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True)

model.train()
loss_fn = nn.MSELoss()

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
final_x, final_y = next(iter(train_loader))
final_x, final_y = final_x.to(device, dtype=torch_dtype), final_y.to(device, dtype=torch_dtype)
with torch.no_grad():
    final_out = model(final_x)
final_prediction = final_out[0, :, 0].to(torch.float32).cpu().numpy()
final_truth = final_y[0, :, 0].to(torch.float32).cpu().numpy()

plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.plot(final_truth, color="#10B981", label="Ground Truth", linewidth=2)
ax.plot(final_prediction, color="#8B5CF6", label="Prediction", linewidth=3)
mse = np.mean((final_prediction - final_truth) ** 2)
ax.fill_between(range(len(final_truth)), final_truth, final_prediction, color="#D1C4E9", alpha=0.9)
ax.set_title(f"Model Predictions vs Ground Truth\nMSE: {mse:.4f}", fontsize=16, pad=15)
ax.set_xlabel("Time Step", fontsize=14)
ax.set_ylabel("Value", fontsize=14)
ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=12)
ax.grid(True, alpha=0.15)
plt.tight_layout(pad=2.0)
plt.savefig("plots/transformer_performance.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nFinal MSE: {mse:.4f}\n")
print("Plot saved to plots/transformer_performance.png")
