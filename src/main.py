from typing import Optional

from fluke.algorithms.dpfedavg import DPFedAVG
from fluke.algorithms.fedavg import FedAVG
from fluke.algorithms.fedprox import FedProx

from src.simulation import run_experiment


def main():
    print("==================================================")
    print("Project: Federated Learning for Medical Diagnosis")
    print("Dataset: Diabetes 130-US Hospitals (1999-2008)")
    print("==================================================")

    # Common settings
    N_CLIENTS = 5
    N_ROUNDS = 10
    BATCH_SIZE = 32
    LR = 0.01
    EPOCHS = 1
    SAMPLE_SIZE: Optional[int] = None

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
        sample_size=SAMPLE_SIZE,
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
        sample_size=SAMPLE_SIZE,
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
        sample_size=SAMPLE_SIZE,
        extra_client_params={"mu": 0.1},  # Proximal term weight
    )

    # 4. Privacy: Differential Privacy with DPFedAVG
    print("\n[Scenario 4] Privacy Preservation - DPFedAVG")
    print("------------------------------------------------")

    # 4.1 Moderate Privacy
    print("Running with Moderate Privacy (Noise Multiplier=1.0)...")
    run_experiment(
        algorithm_class=DPFedAVG,
        distribution="iid",  # Using IID to isolate privacy impact
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
        sample_size=SAMPLE_SIZE,
        extra_client_params={
            "noise_mul": 1.0,
            "max_grad_norm": 1.0,
            "clipping": 1.0,  # Ensure clipping is enabled
        },
    )

    # 4.2 High Privacy
    print("\nRunning with High Privacy (Noise Multiplier=2.0)...")
    run_experiment(
        algorithm_class=DPFedAVG,
        distribution="iid",
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
        sample_size=SAMPLE_SIZE,
        extra_client_params={"noise_mul": 2.0, "max_grad_norm": 1.0, "clipping": 1.0},
    )

    # 5. Fairness & Scalability
    print("\n[Scenario 5] Fairness & Scalability")
    print("-----------------------------------")

    from src.fairness.algorithm import FairFedAVG
    from src.fairness.evaluator import FairnessEvaluator

    # Identify protected attribute index (e.g. 'race' or 'gender')
    # For demonstration, we'll use index 0.
    protected_attr_idx = 0

    # 5.1 Fairness Analysis with Mitigation
    print("\nRunning Fairness Analysis with Mitigation (Lambda=0.5)...")
    fair_evaluator = FairnessEvaluator(
        eval_every=1, n_classes=2, protected_attr_index=protected_attr_idx
    )

    run_experiment(
        algorithm_class=FairFedAVG,
        distribution="iid",
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
        sample_size=SAMPLE_SIZE,
        evaluator=fair_evaluator,  # Inject our custom fairness evaluator
        extra_client_params={
            "fairness_lambda": 0.5  # Strength of fairness regularization
        },
    )

    # 5.2 Scalability Test
    print("\n[Scenario 5.2] Scalability Test (50 Clients, 20% Participation)")
    print("---------------------------------------------------------------")

    # We use standard FedAVG for scalability test, but with many more clients
    run_experiment(
        algorithm_class=FedAVG,
        distribution="iid",
        n_clients=50,  # Large number of clients
        n_rounds=5,  # Reduced rounds for speed in this demo
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        seed=42,
        sample_size=SAMPLE_SIZE,
        eligible_perc=0.2,  # Only 20% of clients (10 clients) participate per round
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
