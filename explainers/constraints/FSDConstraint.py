
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FSDConstraintTorch(nn.Module):
    def __init__(self, lam: float = 1.0, feature_names: list = None, dir_map: dict = None,
                 M: int = None, sample_mode: str = "quantile"):
        """
        lam: float
            Penalty coefficient λ
        feature_names: list
            Order of feature column names
        dir_map: dict
            Direction constraints, e.g. {"Duration": +1}
        M: int
            Number of sampling points (optional, None means full set)
        sample_mode: str
            {"quantile", "random"}
        """
        super().__init__()
        self.lam = lam
        self.feature_names = feature_names if feature_names is not None else []
        self.dir_map = dir_map if dir_map is not None else {}
        self.M = M
        self.sample_mode = sample_mode

    def forward(self, X_fact: torch.Tensor, X_cf: torch.Tensor, mu_list: list) -> torch.Tensor:
        """
        FSD penalty for training phase (supports backpropagation)
        """
        device = X_fact.device
        total_penalty = torch.zeros(1, device=device, dtype=X_fact.dtype)

        # Sample fact if M is set
        if self.M is not None:
            N = X_fact.shape[0]
            if self.sample_mode == "quantile":
                idx = torch.linspace(0, N - 1, self.M).long()
            elif self.sample_mode == "random":
                idx = torch.randperm(N)[:self.M]
            else:
                raise ValueError(f"Unknown sample_mode: {self.sample_mode}")
            X_fact = X_fact[idx]

        for col, direction in self.dir_map.items():
            if col not in self.feature_names:
                continue
            idx = self.feature_names.index(col)

            v_fact = X_fact[:, idx].unsqueeze(1)  # (N,1)
            v_cf   = X_cf[:, idx].unsqueeze(0)    # (1,M)

            # # FSD violation: direction=+1 → cf should shift right
            violation = torch.clamp(direction * (v_fact - v_cf), min=0.0)

            for mu in mu_list:
                mu = mu.to(device)
                penalty = (mu * violation.pow(2)).sum()
                total_penalty = total_penalty + self.lam * penalty

        return total_penalty

    # ================== New unified check interface ==================
    @staticmethod
    def check_FSD(fact, cf, tol=1e-8, direction=+1):
        """
        Unified FSD check interface
        Returns dict including:
            - holds: whether FSD is strictly satisfied (bool)
            - violations: list of violation points
            - point_ratio: proportion of satisfied points
            - interval_ratio: interval-weighted satisfaction proportion
        """
        ecdf_fact = ECDF(fact)
        ecdf_cf   = ECDF(cf)
        critical_points = np.unique(np.concatenate([fact, cf]))

        F_fact = ecdf_fact(critical_points)
        F_cf   = ecdf_cf(critical_points)

        #   # Check point by point
        if direction == +1:
            violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Fc > Ff + tol]
        elif direction == -1:
            violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Ff > Fc + tol]
        else:
            raise ValueError("direction must be +1 (right shift is better) or -1 (left shift is better)")

        holds = (len(violations) == 0)

        # Point ratio
        satisfied_points = len(critical_points) - len(violations)
        point_ratio = satisfied_points / len(critical_points) if len(critical_points) > 0 else 1.0

        # Interval ratio (length-weighted)
        dx = np.diff(critical_points)
        F_diff = (F_cf - F_fact)[:-1]  
        if direction == +1:
            ok_intervals = (F_diff <= tol)
        else:
            ok_intervals = (F_diff >= -tol)
        interval_ratio = dx[ok_intervals].sum() / dx.sum() if dx.sum() > 0 else 1.0

        return {
            "holds": holds,
            "violations": violations,
            "point_ratio": point_ratio,
            "interval_ratio": interval_ratio
        }

    # ================== Keep plotting function unchanged ==================
    @staticmethod
    def plot_cdf(fact, cf, feature_name="", direction=+1):
        ecdf_fact = ECDF(fact)
        ecdf_cf   = ECDF(cf)

        x_min = min(fact.min(), cf.min())
        x_max = max(fact.max(), cf.max())
        x_vals = np.linspace(x_min, x_max, 200)

        plt.figure(figsize=(6,4))
        plt.step(x_vals, ecdf_fact(x_vals), label=f"Factual {feature_name}", color="blue")
        plt.step(x_vals, ecdf_cf(x_vals),   label=f"Counterfactual {feature_name}", color="orange")

        plt.xlabel(feature_name if feature_name else "Feature value")
        plt.ylabel("CDF")
        title_dir = "Right shift is better (+1)" if direction == +1 else "Left shift is better (-1)"
        plt.title(f"CDF Comparison ({feature_name}) [{title_dir}]")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


__all__ = ["FSDConstraintTorch"]

