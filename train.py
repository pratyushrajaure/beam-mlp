import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

df = pd.read_csv("beam_data.csv")

# Splitting dataset
beam_ids = df["beam_id"].unique()
rng = np.random.default_rng(42)
rng.shuffle(beam_ids)

train_ids = beam_ids[:350]
val_ids = beam_ids[350:425]
test_ids = beam_ids[425:500]

train_df = df[df["beam_id"].isin(train_ids)].copy()
val_df = df[df["beam_id"].isin(val_ids)].copy()
test_df = df[df["beam_id"].isin(test_ids)].copy()

# Normalizing the dataset
input_columns = ["x", "L", "E", "I", "P"]
X_train = train_df[input_columns].to_numpy()
X_val = val_df[input_columns].to_numpy()
X_test = test_df[input_columns].to_numpy()
y_train = train_df[["y"]].to_numpy()
y_val = val_df[["y"]].to_numpy()
y_test = test_df[["y"]].to_numpy()

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
if np.any(X_std == 0) or np.any(y_std == 0):
    raise ValueError("Zero standard deviation detected.")

X_train = (X_train - X_mean) / X_std        # normalizing the data
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# dataset creation
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)
val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

class BeamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

model = BeamMLP().to(device)        # model banako

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# training set defining
epochs = 200
best_val_loss = float("inf")
best_epoch = 0
train_losses = []
val_losses = []

# writing the training loop
for epoch in range(epochs):
    # TRAINING
    model.train()
    running_train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)

    train_loss = running_train_loss / len(train_dataset)

    # VALIDATION
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            running_val_loss += loss.item() * inputs.size(0)

    val_loss = running_val_loss / len(val_dataset)

    # SAVE LOSSES
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "X_mean": torch.tensor(X_mean),
            "X_std": torch.tensor(X_std),
            "y_mean": torch.tensor(y_mean),
            "y_std": torch.tensor(y_std)
        }

        torch.save(
            checkpoint,
            "best_model.pth"
        )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

print(f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}")

# loading the best checkpoint
checkpoint = torch.load(
    "best_model.pth",
    map_location=device,
    weights_only=True
)
model = BeamMLP().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Loaded epoch:", checkpoint["epoch"])

# evaluation on the test data
with torch.no_grad():
    predictions_normalized = model(X_test_tensor.to(device)).cpu().numpy()
y_pred = predictions_normalized * y_std + y_mean
y_true = y_test * y_std + y_mean

# metrics
mae_mm = np.mean(np.abs(y_true - y_pred)) * 1000
rmse_mm = np.sqrt(np.mean((y_true - y_pred) ** 2)) * 1000
relative_l2 = (
    np.linalg.norm(y_true - y_pred)/np.linalg.norm(y_true)
)

print("----------------------")
print("\nTest Results")
print("----------------------")
print(f"MAE: {mae_mm:.6f} mm")
print(f"RMSE: {rmse_mm:.6f} mm")
print(f"Relative L2 Error: {relative_l2:.6f}")

# PLOTTING
from pathlib import Path
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(9, 5))
plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Training Loss"
)
plt.plot(
    range(1, epochs + 1),
    val_losses,
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (normalized)")
plt.title("Beam MLP Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_dir / "learning_curve.png",
    dpi=300
)
plt.close()

# predicted vs true plot
y_true_mm = y_true.flatten() * 1000
y_pred_mm = y_pred.flatten() * 1000
plt.figure(figsize=(7, 7))

plt.scatter(
    y_true_mm,
    y_pred_mm,
    alpha=0.5,
    label="Predictions"
)

minimum = min(y_true_mm.min(), y_pred_mm.min())
maximum = max(y_true_mm.max(), y_pred_mm.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)
plt.xlabel("True Deflection (mm)")
plt.ylabel("Predicted Deflection (mm)")
plt.title("Predicted vs True Deflection")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig(
    output_dir / "predicted_vs_true.png",
    dpi=300
)
plt.close()

# error histogram
errors_mm = y_pred_mm - y_true_mm
plt.figure(figsize=(9, 5))
plt.hist(
    errors_mm,
    bins=40,
    edgecolor="black"
)
plt.xlabel("Prediction Error (mm)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    output_dir / "error_histogram.png",
    dpi=300
)
plt.close()

# predicted vs true for three beams
results = test_df.copy()
results["true_y_mm"] = y_true_mm
results["predicted_y_mm"] = y_pred_mm

selected_beams = results["beam_id"].unique()[:3]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

for ax, beam_id in zip(axes, selected_beams):

    beam_data = results[
        results["beam_id"] == beam_id
    ].sort_values("x")

    ax.plot(
        beam_data["x"],
        beam_data["true_y_mm"],
        "o-",
        label="True Deflection"
    )

    ax.plot(
        beam_data["x"],
        beam_data["predicted_y_mm"],
        "s--",
        label="Predicted Deflection"
    )

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_title(f"Test Beam {beam_id}")
    ax.legend()
    ax.grid(True)
    ax.invert_yaxis()
plt.suptitle(
    "Beam Deflection: True vs Predicted"
)
plt.tight_layout()
plt.savefig(
    output_dir / "three_beam_curves.png",
    dpi=300
)
plt.close()