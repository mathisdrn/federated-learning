from src.simulation import run_experiment
from fluke.algorithms.fedavg import FedAVG
from fluke.algorithms.fedprox import FedProx
from fluke import FlukeENV


def main():
    print("==================================================")
    print("Project: Federated Learning for Medical Diagnosis")
    print("Dataset: Heart Disease UCI")
    print("==================================================")

    # Common settings
    N_CLIENTS = 5
    N_ROUNDS = 10
    BATCH_SIZE = 32
    LR = 0.01
    EPOCHS = 1

    # 1. Baseline: IID Data with FedAvg
    print("\n[Scenario 1] IID Data - FedAvg")
    print("------------------------------")
    run_experiment(
        algorithm_class=FedAVG,
        distribution="iid",
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
    )

    # 2. Challenge: Non-IID Data with FedAvg
    # Using Dirichlet distribution for Non-IID skew
    print("\n[Scenario 2] Non-IID Data (Dirichlet Skew) - FedAvg")
    print("---------------------------------------------------")
    run_experiment(
        algorithm_class=FedAVG,
        distribution="dir",
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
    )

    # 3. Treatment: Non-IID Data with FedProx
    print("\n[Scenario 3] Non-IID Data (Dirichlet Skew) - FedProx (Treatment)")
    print("----------------------------------------------------------------")
    run_experiment(
        algorithm_class=FedProx,
        distribution="dir",
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
        extra_client_params={"mu": 0.1},  # Proximal term weight
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
