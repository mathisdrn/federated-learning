from typing import Any, Dict, Optional, Union, Collection
import torch
from torchmetrics import Metric
from fluke.evaluation import Evaluator, ClassificationEval
from fluke.data import FastDataLoader
from src.fairness.metrics import DemographicParity, EqualOpportunity


class FairnessEvaluator(ClassificationEval):
    def __init__(
        self,
        eval_every: int,
        n_classes: int,
        protected_attr_index: int,
        sensitive_group_val: int = 0,
        **metrics: Metric,
    ):
        super().__init__(eval_every, n_classes, **metrics)
        self.protected_attr_index = protected_attr_index
        self.sensitive_group_val = sensitive_group_val

    def evaluate(
        self,
        round: int,
        model: torch.nn.Module,
        eval_data_loader: Union[FastDataLoader, Collection[FastDataLoader]],
        loss_fn: Optional[torch.nn.Module] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Evaluate the model including fairness metrics.
        We override evaluate to ensure inputs (X) are passed to our fairness metrics.
        """

        # Initialize fairness metrics
        dp_metric = DemographicParity(
            self.protected_attr_index, self.sensitive_group_val
        ).to(device)
        eo_metric = EqualOpportunity(
            self.protected_attr_index, self.sensitive_group_val
        ).to(device)

        # Use parent class for standard classification metrics logic?
        # Fluke's Evaluator.evaluate assumes standard metrics (preds, target).
        # Our fairness metrics need (preds, target, inputs).
        # We need to manually run the evaluation loop here or wrap metrics.

        if model is None:
            return {}

        model.eval()
        model.to(device)

        # If it's a collection of loaders (e.g. clients), we might want to aggregate or pick one.
        # Fluke's standard evaluate handles single loader usually for server side.
        loader = eval_data_loader
        if isinstance(eval_data_loader, (list, tuple)):
            # For simplicity in this implementation, if list, just take first or merge
            # But usually server pass a single loader.
            pass

        # Standard metrics from parent
        # We can call super().evaluate() to get standard metrics,
        # but we also need to iterate data for fairness metrics.
        # To avoid double iteration, we will implement the loop here.

        # Reset standard metrics
        for metric in self.metrics.values():
            metric.reset()
            metric.to(device)

        if additional_metrics:
            for metric in additional_metrics.values():
                metric.reset()
                metric.to(device)

        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, (X, y) in enumerate(loader):
                X, y = X.to(device), y.to(device)
                y_hat = model(X)

                if loss_fn:
                    loss = loss_fn(y_hat, y)
                    running_loss += loss.item()

                # Update standard metrics
                # Move predictions to CPU if metrics are on CPU (fluke default)
                # But here we moved metrics to device at start of method.
                # However, fluke's base ClassificationEval seems to move them to cpu for accumulation?
                # Let's check where metrics are. We moved them to device above.
                # So we can keep y_hat on device.

                for metric in self.metrics.values():
                    metric.update(y_hat, y)

                if additional_metrics:
                    for metric in additional_metrics.values():
                        metric.update(y_hat, y)

                # Update Fairness metrics
                dp_metric.update(y_hat, y, X)
                eo_metric.update(y_hat, y, X)

                num_batches += 1

        # Move model back to CPU to free GPU memory and avoid device mismatch later
        model.cpu()
        # Also clean up metrics from GPU if needed, or leave them for compute()

        # Compute results
        results = {}
        if loss_fn:
            results["loss"] = running_loss / max(1, num_batches)

        for name, metric in self.metrics.items():
            results[name] = metric.compute().item()

        if additional_metrics:
            for name, metric in additional_metrics.items():
                results[name] = metric.compute().item()

        # Add fairness results
        results["demographic_parity"] = dp_metric.compute().item()
        results["equal_opportunity"] = eo_metric.compute().item()

        return results
