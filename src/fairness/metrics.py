import torch
from torchmetrics import Metric
from typing import Dict, Any


class DemographicParity(Metric):
    def __init__(self, protected_attr_index: int, sensitive_group_val: int = 0):
        super().__init__()
        self.protected_attr_index = protected_attr_index
        self.sensitive_group_val = sensitive_group_val

        # State variables
        self.add_state(
            "pos_preds_sensitive", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total_sensitive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "pos_preds_others", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total_others", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor
    ) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from the model (logits or probabilities)
            target: Ground truth labels
            inputs: Input features containing the protected attribute
        """
        # Convert logits/probs to binary predictions
        if preds.ndim > 1:
            preds = torch.argmax(preds, dim=1)
        else:
            preds = (preds > 0.5).long()

        # Extract protected attribute column
        # Assuming inputs shape is (batch_size, num_features)
        protected_attr = inputs[:, self.protected_attr_index]

        # Identify sensitive group and others
        # We assume pre-processed data where sensitive group is encoded as sensitive_group_val
        is_sensitive = protected_attr == self.sensitive_group_val
        is_others = ~is_sensitive

        # Update counts
        self.pos_preds_sensitive += torch.sum(preds[is_sensitive])
        self.total_sensitive += torch.sum(is_sensitive)

        self.pos_preds_others += torch.sum(preds[is_others])
        self.total_others += torch.sum(is_others)

    def compute(self) -> float:
        """
        Compute Demographic Parity Difference.
        DP = |P(Y=1 | A=sensitive) - P(Y=1 | A=others)|
        """
        prob_sensitive = self.pos_preds_sensitive.float() / (
            self.total_sensitive.float() + 1e-6
        )
        prob_others = self.pos_preds_others.float() / (self.total_others.float() + 1e-6)

        return torch.abs(prob_sensitive - prob_others)


class EqualOpportunity(Metric):
    def __init__(self, protected_attr_index: int, sensitive_group_val: int = 0):
        super().__init__()
        self.protected_attr_index = protected_attr_index
        self.sensitive_group_val = sensitive_group_val

        # State variables: We only care about instances where Y=1 (target is positive)
        self.add_state(
            "pos_preds_sensitive_y1", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_sensitive_y1", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pos_preds_others_y1", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total_others_y1", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor
    ) -> None:
        # Convert logits/probs to binary predictions
        if preds.ndim > 1:
            preds = torch.argmax(preds, dim=1)
        else:
            preds = (preds > 0.5).long()

        # Extract protected attribute
        protected_attr = inputs[:, self.protected_attr_index]

        # Filter for positive ground truth (Y=1)
        is_y1 = target == 1

        # Combined masks
        is_sensitive_y1 = (protected_attr == self.sensitive_group_val) & is_y1
        is_others_y1 = (protected_attr != self.sensitive_group_val) & is_y1

        # Update counts
        self.pos_preds_sensitive_y1 += torch.sum(preds[is_sensitive_y1])
        self.total_sensitive_y1 += torch.sum(is_sensitive_y1)

        self.pos_preds_others_y1 += torch.sum(preds[is_others_y1])
        self.total_others_y1 += torch.sum(is_others_y1)

    def compute(self) -> float:
        """
        Compute Equal Opportunity Difference.
        EO = |P(Y_hat=1 | A=sensitive, Y=1) - P(Y_hat=1 | A=others, Y=1)|
        This is essentially the difference in True Positive Rates (TPR).
        """
        tpr_sensitive = self.pos_preds_sensitive_y1.float() / (
            self.total_sensitive_y1.float() + 1e-6
        )
        tpr_others = self.pos_preds_others_y1.float() / (
            self.total_others_y1.float() + 1e-6
        )

        return torch.abs(tpr_sensitive - tpr_others)
