import torch
import torch.nn as nn

class MeanConstraintTorch(nn.Module):
    def __init__(self, bounds: dict, lam: float = 1.0, feature_names: list = None):
        """
        bounds: dict
            e.g. {"Duration": (0, 15), "Age": (20, 40)}
        lam: float
            Penalty coefficient λ
        feature_names: list
            The order of feature column names, used to locate indices in the X tensor
        """
        super().__init__()
        self.bounds = bounds
        self.lam = lam
        self.feature_names = feature_names
        self._printed = False  

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        total_penalty = torch.tensor(0.0, device=X.device)

        for col, (low, high) in self.bounds.items():
            idx = self.feature_names.index(col)
            mean_val = X[:, idx].mean()

            penalty_low  = torch.where(mean_val < low,  (low - mean_val) ** 2, torch.tensor(0.0, device=X.device))
            penalty_high = torch.where(mean_val > high, (mean_val - high) ** 2, torch.tensor(0.0, device=X.device))
            penalty = penalty_low + penalty_high

            values = X[:, idx] 
            total_penalty = total_penalty + self.lam * penalty

            # # Print debugging information (optional to enable)
            # print(f"[MeanConstraintTorch] {col}: mean={mean_val.item():.4f}, "
            #       f"low={low}, high={high}, penalty={penalty.item():.4f}")

        return total_penalty


__all__ = ["MeanConstraintTorch"]
