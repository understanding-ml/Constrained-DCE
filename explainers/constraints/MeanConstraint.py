import torch
import torch.nn as nn

class MeanConstraintTorch(nn.Module):
    def __init__(self, bounds: dict, lam: float = 1.0, feature_names: list = None):
        """
        bounds: dict
            例如 {"Duration": (0, 15), "Age": (20, 40)}
        lam: float
            惩罚系数 λ
        feature_names: list
            特征列名的顺序，用于在 X 的 tensor 里定位索引
        """
        super().__init__()
        self.bounds = bounds
        self.lam = lam
        self.feature_names = feature_names
        self._printed = False   # ✅ 初始化标记

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        total_penalty = torch.tensor(0.0, device=X.device)

        for col, (low, high) in self.bounds.items():
            idx = self.feature_names.index(col)
            mean_val = X[:, idx].mean()

            # ✅ 替换为版本1：只针对整体均值做惩罚
            penalty_low  = torch.where(mean_val < low,  (low - mean_val) ** 2, torch.tensor(0.0, device=X.device))
            penalty_high = torch.where(mean_val > high, (mean_val - high) ** 2, torch.tensor(0.0, device=X.device))
            penalty = penalty_low + penalty_high

            # ✅ 保留结构与变量名不变
            values = X[:, idx]  # 仍然保留每个样本的 values，结构不破坏
            total_penalty = total_penalty + self.lam * penalty

            # # 🔎 打印调试信息（可选开启）
            # print(f"[MeanConstraintTorch] {col}: mean={mean_val.item():.4f}, "
            #       f"low={low}, high={high}, penalty={penalty.item():.4f}")

        return total_penalty


__all__ = ["MeanConstraintTorch"]
