"""
Data loader for the Diabetes 130-US Hospitals dataset.

Source: UCI ML Repository (CC BY 4.0)
    Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014).
    https://doi.org/10.24432/C5230J

Target: Binary — readmitted within 30 days (1) vs not (0).

Preprocessing (following Strack et al., 2014):
    1. Remove duplicate patients (keep first encounter)
    2. Remove records where patient died or went to hospice
    3. Drop near-empty columns (weight, payer_code, medical_specialty)
    4. Drop IDs (encounter_id, patient_nbr)
    5. Encode categoricals via ordinal/one-hot encoding
    6. Handle missing values (? → NaN → mode imputation)
    7. Binarize target: '<30' → 1, everything else → 0
    8. StandardScaler on all features

Output: ~71k samples, ~40-50 features after encoding.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _download_if_needed(data_dir):
    """Check if the CSV exists, give instructions if not."""
    csv_path = os.path.join(data_dir, "diabetic_data.csv")
    if os.path.exists(csv_path):
        return csv_path

    # Check for zip
    zip_path = os.path.join(data_dir, "diabetes+130-us+hospitals+for+years+1999-2008.zip")
    if os.path.exists(zip_path):
        print("Extracting dataset from zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        if os.path.exists(csv_path):
            return csv_path

    raise FileNotFoundError(
        f"Dataset not found at {csv_path}\n"
        f"Download from: https://archive.ics.uci.edu/dataset/296\n"
        f"Place diabetic_data.csv (or the zip) in {data_dir}/"
    )


def preprocess_diabetes_hospital(data_dir="data"):
    """
    Full preprocessing pipeline. Returns X (numpy), y (numpy).
    """
    csv_path = _download_if_needed(data_dir)
    print(f"Loading {csv_path}...")

    df = pd.read_csv(csv_path, na_values="?")
    print(f"  Raw: {len(df)} encounters")

    # --- 1. Remove duplicate patients (keep first encounter) ---
    df = df.sort_values("encounter_id")
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    print(f"  After dedup: {len(df)} unique patients")

    # --- 2. Remove expired / hospice patients ---
    # discharge_disposition_id: 11=Expired, 13=Hospice/home, 14=Hospice/medical
    # 19=Expired at home, 20=Expired in medical facility, 21=Expired (place unknown)
    expired_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(expired_ids)]
    print(f"  After removing expired/hospice: {len(df)}")

    # --- 3. Binarize target ---
    # readmitted: '<30' → 1, '>30' or 'NO' → 0
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    print(f"  Readmitted <30 days: {df['readmitted_30'].sum()} "
          f"({df['readmitted_30'].mean()*100:.1f}%)")

    # --- 4. Drop columns ---
    drop_cols = [
        "encounter_id",       # ID
        "patient_nbr",        # ID
        "readmitted",         # original target (replaced by binary)
        "weight",             # >96% missing
        "payer_code",          # >50% missing
        "medical_specialty",  # >49% missing
        "citoglipton",        # single value (no variance)
        "examide",            # single value (no variance)
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- 5. Handle remaining missing values ---
    # race and diag_1/2/3 have some missing
    for col in df.columns:
        if df[col].isnull().any():
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # --- 6. Encode categoricals ---
    # Age bins → ordinal
    age_map = {
        "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
        "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
        "[80-90)": 8, "[90-100)": 9,
    }
    if "age" in df.columns:
        df["age"] = df["age"].map(age_map)

    # A1Cresult and max_glu_serum → ordinal
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].map(a1c_map).fillna(0)

    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].map(glu_map).fillna(0)

    # Medication change columns (Down/Steady/Up/No → numeric)
    med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0)

    # Binary columns
    binary_map = {"No": 0, "Yes": 1, "Ch": 1}
    for col in ["change", "diabetesMed"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    # Gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0)

    # Race — one-hot encode
    if "race" in df.columns:
        df = pd.get_dummies(df, columns=["race"], drop_first=True, dtype=int)

    # Diagnosis codes (diag_1, diag_2, diag_3) — group into categories
    def categorize_diag(val):
        """Group ICD-9 codes into broad categories."""
        if pd.isna(val):
            return 0
        val = str(val)
        if val.startswith("V") or val.startswith("E"):
            return 0
        try:
            code = float(val)
        except ValueError:
            return 0
        if 390 <= code <= 459 or code == 785:
            return 1  # circulatory
        elif 460 <= code <= 519 or code == 786:
            return 2  # respiratory
        elif 520 <= code <= 579 or code == 787:
            return 3  # digestive
        elif 250 <= code < 251:
            return 4  # diabetes
        elif 800 <= code <= 999:
            return 5  # injury
        elif 710 <= code <= 739:
            return 6  # musculoskeletal
        elif 580 <= code <= 629 or code == 788:
            return 7  # genitourinary
        elif 140 <= code <= 239:
            return 8  # neoplasms
        else:
            return 9  # other

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(categorize_diag)

    # --- 7. Final cleanup ---
    # Drop any remaining object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"  Warning: dropping unexpected object columns: {obj_cols}")
        df = df.drop(columns=obj_cols)

    # Separate target
    y = df["readmitted_30"].values
    X = df.drop(columns=["readmitted_30"]).values.astype(np.float32)

    print(f"  Final: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class balance: {y.mean()*100:.1f}% readmitted <30 days")

    return X, y


def load_diabetes_hospital(test_size=None, random_state=42):
    """
    Load preprocessed Diabetes 130-US Hospitals dataset as torch tensors.

    Args:
        test_size: fraction held out (default from config)
        random_state: seed

    Returns:
        X_train, X_test, y_train, y_test as float32 tensors
    """
    from utils.config import TEST_SIZE as DEFAULT_TEST_SIZE

    if test_size is None:
        test_size = DEFAULT_TEST_SIZE

    X, y = preprocess_diabetes_hospital()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Scaler fit on the training split only.
    from utils.data_loader import fit_apply_scaler
    X_train, X_test, _, _ = fit_apply_scaler(X_train, X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def get_raw_diabetes_hospital(scale=True):
    """
    Returns the full dataset as numpy arrays.
    Used by shadow models that need to create their own splits.

    Args:
        scale: if True (default) standardize over the whole file; if False
            return unscaled features so the caller can fit the scaler inside
            its own train/eval split (see utils.data_loader.fit_apply_scaler).

    NOTE: the categorical mode / numeric median imputation inside
    preprocess_diabetes_hospital() is still computed over the whole file. It
    runs before the categorical encoding, so deferring it per split would
    require restructuring that pipeline. At n=69,973 a single record moves a
    column median or mode by a negligible amount, but this remains a
    data-dependent step outside the DP-SGD mechanism and should be disclosed
    as such.
    """
    X, y = preprocess_diabetes_hospital()

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y


if __name__ == "__main__":
    # Quick test
    X, y = preprocess_diabetes_hospital()
    print(f"\nShape: {X.shape}")
    print(f"Target: {y.sum()} positive / {len(y)} total")
