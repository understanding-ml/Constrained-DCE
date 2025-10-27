# import torch
# import torch.nn as nn

# class LSCConstraintTorch(nn.Module):
#     def __init__(self, relation: dict, lam: float = 1.0,
#                  mode: str = "strict", tolerance: float = None):
#         """
#         relation: dict
#             例如 {"Duration": (["Credit amount","Income"], [0.1,0.05], 5.0)}
#             表示 Duration ≈ 0.1*Credit + 0.05*Income + 5
#         lam: float
#             惩罚系数 λ
#         mode: {"strict", "hinge"}
#             strict: 所有偏差都惩罚
#             hinge: 超过 tolerance 才惩罚
#         tolerance: float, optional
#             容忍阈值 (通常设置为 baseline MSE)
#         """
#         super().__init__()
#         self.relation = relation
#         self.lam = lam
#         self.mode = mode
#         self.tolerance = tolerance

#         # 自动提取特征名 → 建立映射 (名字 -> 列号)
#         feature_names = set()
#         for child, (parents, alphas, beta) in relation.items():
#             feature_names.add(child)  # 子变量
#             if isinstance(parents, str):
#                 feature_names.add(parents)  # 单个父变量
#             else:
#                 feature_names.update(parents)  # 多个父变量
#         self.feature_names = sorted(feature_names)  # 统一顺序
#         self.feature_map = {name: i for i, name in enumerate(self.feature_names)}



#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         X: torch.Tensor
#            输入数据 (batch_size × num_features)
#         返回: torch.Tensor (标量, 可反传)
#         """
#         total_penalty = X.new_tensor(0.0)

#         for child, (parents, alphas, beta) in self.relation.items():
#             # 确保父变量统一为 list 形式
#             if isinstance(parents, str):
#                 parents = [parents]
#                 alphas = [alphas]

#             # 找到列索引
#             idx_child = self.feature_map[child]
#             idx_parents = [self.feature_map[p] for p in parents]

#             # 线性预测值
#             pred_child = sum(a * X[:, idx_p] for a, idx_p in zip(alphas, idx_parents)) + beta

#             # 平方差 (不是绝对值)
#             diff = (X[:, idx_child] - pred_child) ** 2  # shape: (batch,)

#             # 根据 mode 决定如何惩罚
#             if self.mode == "strict":
#                 penalty = diff.mean()  # 所有偏差都算
#             elif self.mode == "hinge":
#                 if self.tolerance is None:
#                     raise ValueError("Hinge mode requires tolerance")
#                 penalty = torch.clamp(diff - self.tolerance, min=0).mean()  # 超过阈值才算
#             else:
#                 raise ValueError(f"Unknown mode: {self.mode}")

#             # 累加总惩罚
#             total_penalty = total_penalty + self.lam * penalty

#         return total_penalty


# __all__ = ["LSCConstraintTorch"]




import torch
import torch.nn as nn

class LSCConstraintTorch(nn.Module):
    def __init__(self, relation: dict, lam: float = 1.0,
                 mode: str = "strict", tolerance: float = None,
                 feature_names: list = None):
        """
        relation: dict
            例如 {"Duration": (["Credit amount","Income"], [0.1,0.05], 5.0)}
            表示 Duration ≈ 0.1*Credit + 0.05*Income + 5
        lam: float
            惩罚系数 λ
        mode: {"strict", "hinge"}
            strict: 所有偏差都惩罚
            hinge: 超过 tolerance 才惩罚
        tolerance: float, optional
            容忍阈值 (通常设置为 baseline MSE)
        feature_names: list
            全局特征列名顺序（必须和 X 的列顺序一致）
        """
        super().__init__()
        self.relation = relation
        self.lam = lam
        self.mode = mode
        self.tolerance = tolerance

        if feature_names is None:
            raise ValueError("必须传入 feature_names (全局列名顺序) 才能正确索引张量列！")
        self.feature_names = feature_names

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: torch.Tensor
           输入数据 (batch_size × num_features)
        返回: torch.Tensor (标量, 可反传)
        """
        total_penalty = X.new_tensor(0.0)

        for child, (parents, alphas, beta) in self.relation.items():
            # 确保父变量统一为 list
            if isinstance(parents, str):
                parents = [parents]
                alphas = [alphas]

            # 找到列索引（严格根据传入的 feature_names，不会乱序）
            idx_child = self.feature_names.index(child)
            idx_parents = [self.feature_names.index(p) for p in parents]

            # 线性预测值
            pred_child = sum(a * X[:, idx_p] for a, idx_p in zip(alphas, idx_parents)) + beta

            # 平方差
            diff = (X[:, idx_child] - pred_child) ** 2  # shape: (batch,)

            # 根据 mode 决定如何惩罚
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
        单独计算每个子变量的 MSE（不加 λ、不加 hinge）。
        返回 dict: {child: mse_value}
        """
        mse_results = {}
        for child, (parents, alphas, beta) in self.relation.items():
            if isinstance(parents, str):
                parents, alphas = [parents], [alphas]

            idx_child = self.feature_names.index(child)
            idx_parents = [self.feature_names.index(p) for p in parents]

            pred_child = sum(a * X[:, idx_p] for a, idx_p in zip(alphas, idx_parents)) + beta
            diff = (X[:, idx_child] - pred_child) ** 2
            mse_results[child] = diff.mean().item()  # 每个子变量的 MSE
        return mse_results

__all__ = ["LSCConstraintTorch"]
