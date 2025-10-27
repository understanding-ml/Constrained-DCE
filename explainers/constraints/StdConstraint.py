import torch
import torch.nn as nn

class StdConstraintTorch(nn.Module):
    def __init__(self, bounds: dict, lam: float = 1.0, feature_names: list = None):
        """
        bounds: dict
            例如 {"Duration": 5.0, "Age": 10.0}
            表示要求某个特征的标准差 <= bound
        lam: float
            惩罚系数 λ
        feature_names: list
            特征列名的顺序，用于在 X 的 tensor 里定位索引
        """
        super().__init__()
        # 内部统一转成方差上界
        self.var_bounds = {col: bound ** 2 for col, bound in bounds.items()}
        self.lam = lam
        self.feature_names = feature_names
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        total_penalty = X.new_tensor(0.0)  # 保证 dtype/device 一致

        for col, var_up in self.var_bounds.items():
            idx = self.feature_names.index(col)

            # 经验均值 μ_d
            mean_val = X[:, idx].mean()
            # 经验方差 Var_d
            var_val = ((X[:, idx] - mean_val) ** 2).mean()

            # Hinge 惩罚: max(0, Var_d - σ^2_up)^2
            penalty = torch.clamp(var_val - var_up, min=0) ** 2

            total_penalty = total_penalty + self.lam * penalty

            # 🔎 Debug（按需打开）
            # print(f"[StdConstraintTorch] {col}: std={torch.sqrt(var_val).item():.4f}, "
            #       f"up={var_up**0.5}, penalty={penalty.item():.4f}")
            
        return total_penalty


__all__ = ["StdConstraintTorch"]
