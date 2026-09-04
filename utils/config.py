"""
Shared configuration for the FL privacy project.
All scripts import from here so parameters stay consistent.

Change DATASET to switch between experiments.
"""

# --- Dataset selection ---
# Options: "breast_cancer", "heart_disease", "cardiotocography", "diabetes_hospital"
DATASET = "diabetes_hospital"


# --- Data split ---
# Per-dataset recommended splits:
#   breast_cancer:      0.95 (5% training, ~28 samples — stress test)
#   heart_disease:      0.90 (10% training, ~92 samples)
#   cardiotocography:   0.85 (15% training, ~319 samples)
#   diabetes_hospital:  0.85 (15% training, ~10,495 samples)
DATASET_SPLITS = {
    "breast_cancer": 0.95,
    "heart_disease": 0.90,
    "cardiotocography": 0.85,
    "diabetes_hospital": 0.85,
}
TEST_SIZE = DATASET_SPLITS.get(DATASET, 0.85)

# Training objective
OBJECTIVE = "weighted"


VAL_FRACTIONS = {
    "heart_disease": 0.30,      # 920 samples, 10% train -> ~64 train / ~28 val
    "cardiotocography": 0.20,   # 2,126 samples, 15% train -> ~254 / ~64
    "breast_cancer": 0.20,      # 569 at 5% train -> 22 / 6: fails the guard,
                                # so this dataset is stress-test only
    "diabetes_hospital": 0.20,  # ample either way
}
VAL_FRACTION = VAL_FRACTIONS.get(DATASET, 0.20)

# --- Model ---
INPUT_DIM = None  # set dynamically based on dataset

# --- Training ---
LR = 0.001
BATCH_SIZE = 16

# --- DP-SGD ---
NOISE_MULTIPLIER = 0.8
MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5

# --- Target model training ---
TARGET_EPOCHS = 50

# --- Shadow models ---
NUM_SHADOW_MODELS = 15
SHADOW_EPOCHS_NO_DP = 30
SHADOW_EPOCHS_DP = 50

# --- Attack model ---
ATTACK_NN_EPOCHS = 200
