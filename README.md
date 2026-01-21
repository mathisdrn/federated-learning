# Federated Learning for Medical Diagnosis

This project implements a Federated Learning (FL) system for medical diagnosis using the **Fluke** framework. It explores the challenges of heterogeneity (Non-IID data) and privacy in a distributed medical setting.

## Project Structure

```
.
├── src/
│   ├── main.py         # Entry point for the experiments
│   ├── dataset.py      # Data loading, preprocessing, and download logic
│   ├── models.py       # PyTorch model architecture (BinaryClassifier)
│   └── simulation.py   # Reusable FL experiment logic (FedAvg, FedProx)
├── diabetic_data.csv   # Dataset Diabetes 130-US hospitals
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # This file
```

## Setup

This project uses `uv` for dependency management.

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```
    This will install python 3.12 and all required libraries (fluke-fl, torch, pandas, etc.).

## Running Experiments

To run the full suite of experiments (IID vs Non-IID vs FedProx), execute:

```bash
uv run -m src.main
```

### Experiment Scenarios

The script (`src/main.py`) runs three scenarios using the **Dataset Diabetes 130-US hospitals** dataset:

1.  **Baseline (IID)**:
    *   **Algorithm**: FedAvg
    *   **Distribution**: IID (Uniform)
    *   **Result**: High accuracy (~77%) establishes a baseline.

2.  **Challenge (Non-IID)**:
    *   **Algorithm**: FedAvg
    *   **Distribution**: Non-IID (Dirichlet Skew, $\beta=0.5$)
    *   **Result**: Performance drops significantly (~55%) due to data heterogeneity across clients.

3.  **Treatment (FedProx)**:
    *   **Algorithm**: FedProx
    *   **Distribution**: Non-IID (Dirichlet Skew, $\beta=0.5$)
    *   **Parameter**: $\mu=0.1$ (Proximal term)
    *   **Goal**: Improve convergence and stability in heterogeneous settings.

## Data

The project uses the **Diabetes 130-US Hospitals** dataset.
*   **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
*   **Preprocessing**:
    *   Downloads automatically if missing via `src/dataset.py`.
    *   Converts targets to Binary Classification (Presence vs Absence of heart disease).
    *   Normalizes features (StandardScaler).
    *   Splits into train/test sets for clients and server.