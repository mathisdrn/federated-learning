import torch.nn as nn
from fluke import DDict, FlukeENV
from fluke.algorithms import CentralizedFL
from fluke.algorithms.fedavg import FedAVG
from fluke.data import DataSplitter
from fluke.evaluation import ClassificationEval

from src.dataset import get_fluke_dataset
from src.models import BinaryClassifier


def run_experiment(
    algorithm_class: type[CentralizedFL] = FedAVG,
    distribution="iid",
    n_clients=5,
    n_rounds=10,
    batch_size=32,
    lr=0.01,
    epochs=1,
    seed=42,
    extra_client_params=None,
    sample_size=None,
):
    # 1. Setup Environment
    # Re-instantiating FlukeENV singleton to update settings if needed
    env = FlukeENV()
    env.set_seed(seed)
    env.set_device("auto")

    # Configure Evaluator
    evaluator = ClassificationEval(eval_every=1, n_classes=2)
    env.set_evaluator(evaluator)

    # 2. Prepare Data
    print("Loading data...")
    data_container, input_dim = get_fluke_dataset(
        batch_size=batch_size, sample_size=sample_size
    )

    # 3. Create Data Splitter
    print(f"Splitting data ({distribution})...")

    # Configure distribution arguments if needed (e.g. for Dirichlet)
    dist_args = None
    if distribution == "dir":
        dist_args = DDict(beta=0.5, min_ex_class=1, balanced=False)

    splitter = DataSplitter(
        dataset=data_container,
        distribution=distribution,
        server_test=True,
        keep_test=True,
        client_split=0.2,
        dist_args=dist_args or DDict(),
    )

    # 4. Define Model
    model = BinaryClassifier(input_dim=input_dim)

    # 5. Configure Hyperparameters
    client_config = DDict(
        batch_size=batch_size,
        epochs=epochs,
        local_epochs=epochs,  # Explicitly adding local_epochs for clients like FedProx
        optimizer=DDict(name="SGD", lr=lr, momentum=0.9),
        scheduler=None,
        loss=nn.CrossEntropyLoss,
    )

    if extra_client_params:
        client_config.update(extra_client_params)

    hyper_params = DDict(model=model, client=client_config, server=DDict(weighted=True))

    # 6. Initialize Algorithm
    algo_name = algorithm_class.__name__
    print(f"Initializing {algo_name} with {n_clients} clients...")
    algo = algorithm_class(
        n_clients=n_clients, data_splitter=splitter, hyper_params=hyper_params
    )

    # 7. Run Experiment
    print(f"Starting training for {n_rounds} rounds...")
    # Capture output or logs if possible, but Fluke prints progress bars
    algo.run(n_rounds=n_rounds, eligible_perc=1.0)

    # 8. Final Evaluation
    print("Evaluating final model...")
    metrics = algo.server.evaluate(evaluator, algo.server.test_set)
    print(f"Final Global Metrics: {metrics}")

    print(f"{algo_name} Experiment finished.")
    return algo


if __name__ == "__main__":
    run_experiment(FedAVG)
