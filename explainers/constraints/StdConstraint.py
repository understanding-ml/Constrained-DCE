import torch
import torch.nn as nn

class StdConstraintTorch(nn.Module):
    def __init__(self, bounds: dict, lam: float = 1.0, feature_names: list = None):
        """
        bounds: dict
            e.g. {"Duration": 5.0, "Age": 10.0}
            Indicates that the standard deviation of a feature is required to be <= bound
        lam: float
            Penalty coefficient λ
        feature_names: list
            The order of feature column names, used to locate indices in the X tensor
        """
        super().__init__()
        # Uniformly convert to variance upper bound internally
        self.var_bounds = {col: bound ** 2 for col, bound in bounds.items()}
        self.lam = lam
        self.feature_names = feature_names
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        total_penalty = X.new_tensor(0.0)  

        for col, var_up in self.var_bounds.items():
            idx = self.feature_names.index(col)

            # Empirical mean μ_d
            mean_val = X[:, idx].mean()
            # Empirical variance Var_d
            var_val = ((X[:, idx] - mean_val) ** 2).mean()

            # Hinge penalty: max(0, Var_d - σ^2_up)^2
            penalty = torch.clamp(var_val - var_up, min=0) ** 2

            total_penalty = total_penalty + self.lam * penalty

            # # Debug (enable as needed)
            # print(f"[StdConstraintTorch] {col}: std={torch.sqrt(var_val).item():.4f}, "
            #       f"up={var_up**0.5}, penalty={penalty.item():.4f}")
            
        return total_penalty


__all__ = ["StdConstraintTorch"]
