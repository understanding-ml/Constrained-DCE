import torch
import torch.nn as nn

class LSCConstraintTorch(nn.Module):
    def __init__(self, relation: dict, lam: float = 1.0,
                 mode: str = "strict", tolerance: float = None,
                 feature_names: list = None):
        """
        relation: dict
            e.g. {"Duration": (["Credit amount","Income"], [0.1,0.05], 5.0)}
            Represents Duration ≈ 0.1*Credit + 0.05*Income + 5
        lam: float
            Penalty coefficient λ
        mode: {"strict", "hinge"}
            strict: Penalize all deviations
            hinge: Only penalize deviations exceeding tolerance
        tolerance: float, optional
            Tolerance threshold (usually set to baseline MSE)
        feature_names: list
            Global order of feature column names (must match the column order of X)
        """
        super().__init__()
        self.relation = relation
        self.lam = lam
        self.mode = mode
        self.tolerance = tolerance

        if feature_names is None:
            raise ValueError("feature_names (global column name order) must be provided to index tensor columns correctly!")
        self.feature_names = feature_names

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: torch.Tensor
           Input data (batch_size × num_features)
        Returns: torch.Tensor (scalar, supports backpropagation)
        """
        total_penalty = X.new_tensor(0.0)

        for child, (parents, alphas, beta) in self.relation.items():
            # Ensure parent variables are uniformly in list format
            if isinstance(parents, str):
                parents = [parents]
                alphas = [alphas]

            # Find column indices (strictly based on the input feature_names, no disorder)
            idx_child = self.feature_names.index(child)
            idx_parents = [self.feature_names.index(p) for p in parents]

            # Linear predicted value
            pred_child = sum(a * X[:, idx_p] for a, idx_p in zip(alphas, idx_parents)) + beta

            # Squared error
            diff = (X[:, idx_child] - pred_child) ** 2  # shape: (batch,)

            # Determine penalty method based on mode
            if self.mode == "strict":
                penalty = diff.mean()
            elif self.mode == "hinge":
                if self.tolerance is None:
                    raise ValueError("Hinge mode requires tolerance")
                penalty = torch.clamp(diff - self.tolerance, min=0).mean()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            total_penalty = total_penalty + self.lam * penalty

        return total_penalty

    def compute_mse(self, X: torch.Tensor) -> dict:
        """
        Calculate MSE for each child variable independently (without λ or hinge penalty).
        Returns dict: {child: mse_value}
        """
        mse_results = {}
        for child, (parents, alphas, beta) in self.relation.items():
            if isinstance(parents, str):
                parents, alphas = [parents], [alphas]

            idx_child = self.feature_names.index(child)
            idx_parents = [self.feature_names.index(p) for p in parents]

            pred_child = sum(a * X[:, idx_p] for a, idx_p in zip(alphas, idx_parents)) + beta
            diff = (X[:, idx_child] - pred_child) ** 2
            mse_results[child] = diff.mean().item()   # MSE for each child variable
        return mse_results

__all__ = ["LSCConstraintTorch"]
