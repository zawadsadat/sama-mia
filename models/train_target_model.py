"""
Train the target model with DP-SGD (Opacus).

This is the model that the MIA will try to attack.
Deliberately overparameterized + small training set = memorization.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from models.model import TargetModel
from utils.data_loader import load_data, get_input_dim, get_dataset_info
from utils.utility_metrics import evaluate_utility, format_utility
from utils.objective import make_criterion
from utils.config import (
    OBJECTIVE,
    TARGET_EPOCHS as EPOCHS, BATCH_SIZE, LR,
    NOISE_MULTIPLIER, MAX_GRAD_NORM, DP_DELTA,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset info ---
info = get_dataset_info()
input_dim = get_input_dim()

print(f"Dataset: {info['name']}")
print(f"  Total samples: {info['samples']}, Features: {info['features']}")
print(f"  Positive rate: {info['positive_rate']*100:.1f}%")
print(f"  Input dim: {input_dim}")
print()

# --- Data ---
X_train, X_test, y_train, y_test = load_data()
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model + DP ---
model = TargetModel(input_dim=input_dim).to(device)
criterion, pos_w = make_criterion(y_train, OBJECTIVE)
print(f"Objective: {OBJECTIVE} (pos_weight={pos_w:.3f})")
optimizer = optim.Adam(model.parameters(), lr=LR)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
)

# --- Training ---
for epoch in range(EPOCHS):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# --- Evaluation ---
# Reviewer comment 4: plain accuracy is achievable by a constant majority-class
# predictor on imbalanced data (Diabetes 9% positive). evaluate_utility reports
# prevalence-robust metrics and flags degenerate (constant) predictors.
# NOTE: TargetModel ends in nn.Sigmoid(), so model(X) is already a probability.
# If the model is ever switched to BCEWithLogitsLoss, wrap this in torch.sigmoid().
with torch.no_grad():
    probs = model(X_test).squeeze().cpu().numpy()

utility = evaluate_utility(probs, y_test.cpu().numpy())
accuracy = utility["acc"]

print("\n--- Target task utility ---")
print(format_utility(utility))
if utility["degenerate"]:
    print("\n  !! Model predicts a single class for every sample.")
    print("     Accuracy here equals the majority-class baseline and carries")
    print("     no information. Any MIA/FMPAS result on this model reflects")
    print("     the absence of a model, not the presence of privacy.")

epsilon = privacy_engine.get_epsilon(delta=DP_DELTA)
print(f"Privacy budget: ε = {epsilon:.2f}, δ = {DP_DELTA}")

# --- Save ---
os.makedirs("experiments", exist_ok=True)
torch.save(model.state_dict(), "experiments/target_model.pt")

# Save input_dim for later loading
torch.save({"input_dim": input_dim}, "experiments/model_config.pt")

# Persist utility so downstream scripts can apply the FMPAS utility floor
# without retraining.
import json
_tag = "dp" if NOISE_MULTIPLIER else "nodp"
with open(f"experiments/target_utility_{info['name'].replace(' ', '_')}_{_tag}.json", "w") as f:
    json.dump({"epsilon": epsilon, "noise_multiplier": NOISE_MULTIPLIER, **utility}, f, indent=2)

print("Target model saved.")
