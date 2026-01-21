import torch
import torch.nn as nn
from fluke.client import Client
from fluke.data import FastDataLoader
from fluke.config import OptimizerConfigurator
from fluke import DDict
from fluke.utils import clear_cuda_cache


class FairClient(Client):
    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: nn.Module,
        local_epochs: int,
        fairness_lambda: float = 0.0,
        protected_attr_index: int = 0,
        sensitive_group_val: int = 0,
        **kwargs,
    ):
        super().__init__(
            index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs, **kwargs
        )
        self.hyper_params.update(
            fairness_lambda=fairness_lambda,
            protected_attr_index=protected_attr_index,
            sensitive_group_val=sensitive_group_val,
        )

    def _fairness_regularization(
        self, preds: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a regularization term to penalize unfair predictions.
        Simple approach: Correlation penalty between predictions and protected attribute.
        """
        # Extract protected attribute
        protected_attr = inputs[:, self.hyper_params.protected_attr_index].float()

        # We want to minimize correlation between positive prediction probability and sensitive attribute
        # Prob(y=1)
        if preds.shape[1] > 1:
            probs = torch.softmax(preds, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(preds).squeeze()

        # Covariance(probs, protected_attr)
        probs_mean = torch.mean(probs)
        attr_mean = torch.mean(protected_attr)

        covariance = torch.mean((probs - probs_mean) * (protected_attr - attr_mean))

        # We penalize the absolute covariance
        return torch.abs(covariance)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs = (
            override_local_epochs
            if override_local_epochs
            else self.hyper_params.local_epochs
        )

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0

        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                y_hat = self.model(X)

                # Standard Task Loss
                task_loss = self.hyper_params.loss_fn(y_hat, y)

                # Fairness Regularization
                fair_reg = 0.0
                if self.hyper_params.fairness_lambda > 0:
                    fair_reg = self._fairness_regularization(y_hat, X)

                total_loss = task_loss + (self.hyper_params.fairness_lambda * fair_reg)

                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()

            if self.scheduler:
                self.scheduler.step()

        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        clear_cuda_cache()
        return running_loss
