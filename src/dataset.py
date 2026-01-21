import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
from fluke.data import DataContainer
from ucimlrepo import fetch_ucirepo


class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # CrossEntropyLoss expects long targets
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def download_heart_disease(save_path="data/raw/heart_disease.csv"):
    print("Fetching Heart Disease UCI dataset...")

    # fetch dataset
    # ID 45 is the standard Heart Disease dataset
    try:
        heart_disease = fetch_ucirepo(id=45)

        # data (as pandas dataframes)
        X = heart_disease.data.features
        y = heart_disease.data.targets

        # Combine features and target
        df = pd.concat([X, y], axis=1)

        # Save to CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"Dataset saved to {save_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        raise


def load_and_preprocess_data(
    filepath="data/raw/heart_disease.csv", test_size=0.2, seed=42
):
    """
    Loads the Heart Disease dataset, handles missing values, normalizes features,
    and splits into train and test sets.
    """
    if not os.path.exists(filepath):
        # Fallback to verify if we are in root
        if os.path.exists(os.path.join("..", filepath)):
            filepath = os.path.join("..", filepath)
        else:
            print(f"Dataset not found at {filepath}. Attempting download...")
            download_heart_disease(filepath)

    df = pd.read_csv(filepath)

    # The target is 'num' (diagnosis of heart disease).
    # value 0: < 50% diameter narrowing (no disease)
    # value 1-4: > 50% diameter narrowing (disease)
    # We convert this to a binary classification task.
    if "num" in df.columns:
        target_col = "num"
    elif "target" in df.columns:  # Sometimes UCI datasets rename it
        target_col = "target"
    else:
        # Fallback based on typical column position if name differs
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    # Ensure binary target
    y = df[target_col].apply(lambda x: 1 if x > 0 else 0).values

    # Handle missing values
    # Some columns in this dataset might have '?' or NaN
    X.replace("?", np.nan, inplace=True)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Normalize/Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    # For CrossEntropy, targets should be long (N,)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def get_fluke_dataset(filepath="data/raw/heart_disease.csv", batch_size=32):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Create Fluke DataContainer
    # num_classes = 2 for binary classification
    data_container = DataContainer(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_classes=2
    )

    return data_container, X_train.shape[1]
