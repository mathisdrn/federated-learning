from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fluke.data import DataContainer


DIABETES_FILE = "diabetic_data.csv"
DEFAULT_SAMPLE_SIZE = 1000
DROP_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
]
MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
CATEGORICAL_COLUMNS = [
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "change",
    "diabetesMed",
    "max_glu_serum",
    "A1Cresult",
]
SCALER_COLUMNS = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
GENDER_MAP = {"Female": 0, "Male": 1}
MEDICATION_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
TARGET_MAP = {"NO": 0, "<30": 1, ">30": 1}
AGE_MAP = {
    "[0-10)": 0,
    "[10-20)": 1,
    "[20-30)": 2,
    "[30-40)": 3,
    "[40-50)": 4,
    "[50-60)": 5,
    "[60-70)": 6,
    "[70-80)": 7,
    "[80-90)": 8,
    "[90-100)": 9,
}


def _resolve_filepath(filepath: str) -> Path:
    path = Path(filepath)
    if path.is_file():
        return path

    project_root = Path(__file__).resolve().parent.parent
    alt_path = project_root / filepath
    if alt_path.is_file():
        return alt_path

    raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")


def _load_diabetes_dataframe(filepath: str) -> pd.DataFrame:
    path = _resolve_filepath(filepath)
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    if existing_drop_cols:
        df.drop(columns=existing_drop_cols, inplace=True)

    if "gender" in df.columns:
        df = df.loc[df["gender"].ne("Unknown/Invalid")].copy()

    required_cols = [col for col in ["race", "gender", "age"] if col in df.columns]
    if required_cols:
        df.dropna(subset=required_cols, inplace=True)

    if "age" in df.columns:
        df["age"] = df["age"].replace(AGE_MAP)

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace(GENDER_MAP)

    medication_cols = [col for col in MEDICATION_COLUMNS if col in df.columns]
    for col in medication_cols:
        df[col] = df[col].replace(MEDICATION_MAP).fillna(0).astype(int)

    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    if cat_cols:
        for col in cat_cols:
            df[col] = df[col].fillna("missing").astype(str)
        dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=np.float32)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)

    return df


def load_and_preprocess_data(
    filepath: str = DIABETES_FILE,
    test_size: float = 0.2,
    seed: int = 42,
    sample_size: Optional[int] = None,
):
    df = _load_diabetes_dataframe(filepath)
    df = _prepare_features(df)

    if "readmitted" not in df.columns:
        raise ValueError("Column 'readmitted' is required in the dataset.")

    allowed_targets = list(TARGET_MAP.keys())
    df = df[df["readmitted"].isin(allowed_targets)]

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    y = df.pop("readmitted").replace(TARGET_MAP).astype(int)
    X = df

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    X_train = pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True)
    X_test = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    imputer = SimpleImputer(strategy="mean")
    X_train_np = imputer.fit_transform(X_train.to_numpy(dtype=np.float32, copy=True))
    X_test_np = imputer.transform(X_test.to_numpy(dtype=np.float32, copy=True))

    col_index = {col: idx for idx, col in enumerate(X_train.columns)}
    scale_indices = np.array(
        [col_index[col] for col in SCALER_COLUMNS if col in col_index]
    )
    if scale_indices.size > 0:
        scaler = StandardScaler()
        X_train_np[:, scale_indices] = scaler.fit_transform(
            X_train_np[:, scale_indices]
        )
        X_test_np[:, scale_indices] = scaler.transform(X_test_np[:, scale_indices])

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def get_fluke_dataset(
    filepath: str = DIABETES_FILE,
    batch_size: int = 32,
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        filepath=filepath, sample_size=sample_size
    )

    data_container = DataContainer(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_classes=2
    )

    return data_container, X_train.shape[1]
