# # import numpy as np
# # from statsmodels.distributions.empirical_distribution import ECDF
# # import torch
# # import torch.nn as nn

# # class FSDConstraintTorch(nn.Module):
# #     def __init__(self, lam: float = 1.0, feature_names: list = None, dir_map: dict = None):
# #         """
# #         lam: float
# #             惩罚系数 λ
# #         feature_names: list
# #             特征列名顺序，用于在 torch.Tensor 里找到对应索引
# #         dir_map: dict
# #             方向约束，例如 {"Duration": +1}
# #             +1 表示该特征在 cf 中应整体右移 (cf 更大 → 更优，比如贷款期限更长)
# #             -1 表示该特征在 cf 中应整体左移 (cf 更小 → 更优，比如等待时间更短)
# #         """
# #         super().__init__()
# #         self.lam = lam
# #         self.feature_names = feature_names if feature_names is not None else []
# #         self.dir_map = dir_map if dir_map is not None else {}

# #     def forward(self, X_fact: torch.Tensor, X_cf: torch.Tensor, mu_list: list) -> torch.Tensor:
# #         """
# #         X_fact: torch.Tensor, shape (N, D)
# #             原始分布的样本 (fact)
# #         X_cf: torch.Tensor, shape (M, D)
# #             反事实分布的样本 (counterfactual)
# #         mu_list: list of torch.Tensor
# #             Wasserstein 对齐后的传输矩阵 (N×M)，可来自 Manager 逐维计算

# #         返回:
# #         total_penalty: torch.Tensor
# #             满足 FSD 的惩罚值 (scalar, 可反向传播)
# #         """
# #         device = X_fact.device
# #         total_penalty = torch.zeros(1, device=device, dtype=X_fact.dtype)

# #         for col, direction in self.dir_map.items():
# #             if col not in self.feature_names:
# #                 continue
# #             idx = self.feature_names.index(col)

# #             v_fact = X_fact[:, idx].unsqueeze(1)  # (N,1)
# #             v_cf   = X_cf[:, idx].unsqueeze(0)    # (1,M)

# #             # 如果不满足 FSD (direction=+1 → cf 应该右移)，差值为正才算违规
# #             violation = torch.clamp(direction * (v_fact - v_cf), min=0.0)

# #             for mu in mu_list:
# #                 mu = mu.to(device)
# #                 penalty = (mu * violation.pow(2)).sum()
# #                 total_penalty = total_penalty + self.lam * penalty

# #             # 🔎 调试信息 (可选)
# #             # print(f"[FSDConstraintTorch] {col} (dir={direction}): penalty={penalty.item():.4f}")

# #         return total_penalty

# #     @staticmethod
# #     def check_FSD(fact, cf, tol=1e-8, return_violations=False):
# #         """
# #         检查一维分布 cf 是否在 ECDF 意义下满足对 fact 的一阶随机占优。
# #         用法示例:
# #             fact = np.array([1,2,3,4,5])
# #             cf   = np.array([2,3,4,5,6])   # cf 向右移 → 应该满足 FSD
# #             FSDConstraintTorch.check_FSD(fact, cf)
# #         """
# #         ecdf_fact = ECDF(fact)
# #         ecdf_cf   = ECDF(cf)
# #         critical_points = np.unique(np.concatenate([fact, cf]))

# #         F_fact = ecdf_fact(critical_points)
# #         F_cf   = ecdf_cf(critical_points)

# #         violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Fc > Ff + tol]
# #         fsd_holds = (len(violations) == 0)

# #         return (fsd_holds, violations) if return_violations else fsd_holds


# # __all__ = ["FSDConstraintTorch"]





# #_________0928引入采样比例————————————————————

# import numpy as np
# from statsmodels.distributions.empirical_distribution import ECDF
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# class FSDConstraintTorch(nn.Module):
#     def __init__(self, lam: float = 1.0, feature_names: list = None, dir_map: dict = None,
#                  M: int = None, sample_mode: str = "quantile"):
#         """
#         lam: float
#             惩罚系数 λ
#         feature_names: list
#             特征列名顺序
#         dir_map: dict
#             方向约束，例如 {"Duration": +1}
#         M: int
#             采样点数（可选，None 表示全量）
#         sample_mode: str
#             采样方式 {"quantile", "random"}
#         """
#         super().__init__()
#         self.lam = lam
#         self.feature_names = feature_names if feature_names is not None else []
#         self.dir_map = dir_map if dir_map is not None else {}
#         self.M = M
#         self.sample_mode = sample_mode

#     def forward(self, X_fact: torch.Tensor, X_cf: torch.Tensor, mu_list: list) -> torch.Tensor:
#         device = X_fact.device
#         total_penalty = torch.zeros(1, device=device, dtype=X_fact.dtype)

#         # 🚩 如果设置了 M，对 fact 做采样
#         if self.M is not None:
#             N = X_fact.shape[0]
#             if self.sample_mode == "quantile":
#                 idx = torch.linspace(0, N - 1, self.M).long()
#             elif self.sample_mode == "random":
#                 idx = torch.randperm(N)[:self.M]
#             else:
#                 raise ValueError(f"Unknown sample_mode: {self.sample_mode}")
#             X_fact = X_fact[idx]

#         for col, direction in self.dir_map.items():
#             if col not in self.feature_names:
#                 continue
#             idx = self.feature_names.index(col)

#             v_fact = X_fact[:, idx].unsqueeze(1)  # (N,1)
#             v_cf   = X_cf[:, idx].unsqueeze(0)    # (1,M)

#             # FSD 违规：direction=+1 → cf 应该右移
#             violation = torch.clamp(direction * (v_fact - v_cf), min=0.0)

#             for mu in mu_list:
#                 mu = mu.to(device)
#                 penalty = (mu * violation.pow(2)).sum()
#                 total_penalty = total_penalty + self.lam * penalty

#         return total_penalty

#     @staticmethod
#     def check_FSD(fact, cf, tol=1e-8, return_violations=False, direction=+1):
#         """
#         检查 cf 是否在 ECDF 意义下满足对 fact 的一阶随机占优。
#         direction=+1 → cf 应整体右移
#         direction=-1 → cf 应整体左移
#         """
#         ecdf_fact = ECDF(fact)
#         ecdf_cf   = ECDF(cf)
#         critical_points = np.unique(np.concatenate([fact, cf]))

#         F_fact = ecdf_fact(critical_points)
#         F_cf   = ecdf_cf(critical_points)

#         if direction == +1:
#             violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Fc > Ff + tol]
#         elif direction == -1:
#             violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Ff > Fc + tol]
#         else:
#             raise ValueError("direction must be +1 (右移) or -1 (左移)")

#         fsd_holds = (len(violations) == 0)
#         return (fsd_holds, violations) if return_violations else fsd_holds
    
#      # 🚩 新增：画 Factual vs Counterfactual CDF
#     @staticmethod
#     def plot_cdf(fact, cf, feature_name="", direction=+1):
#         """
#         画出 Factual vs Counterfactual 的 CDF 曲线
#         参数:
#             fact: numpy array, 事实分布
#             cf: numpy array, 反事实分布
#             feature_name: 特征名，用于标签
#             direction: +1 → cf 应右移更优, -1 → cf 应左移更优
#         """
#         ecdf_fact = ECDF(fact)
#         ecdf_cf   = ECDF(cf)

#         x_min = min(fact.min(), cf.min())
#         x_max = max(fact.max(), cf.max())
#         x_vals = np.linspace(x_min, x_max, 200)

#         plt.figure(figsize=(6,4))
#         plt.step(x_vals, ecdf_fact(x_vals), label=f"Factual {feature_name}", color="blue")
#         plt.step(x_vals, ecdf_cf(x_vals),   label=f"Counterfactual {feature_name}", color="orange")

#         plt.xlabel(feature_name if feature_name else "Feature value")
#         plt.ylabel("CDF")

#         # 根据方向显示对比逻辑
#         title_dir = "Right shift is better (+1)" if direction == +1 else "Left shift is better (-1)"
#         plt.title(f"CDF Comparison ({feature_name}) [{title_dir}]")

#         plt.legend()
#         plt.grid(True, linestyle="--", alpha=0.7)
#         plt.tight_layout()
#         plt.show()


#     @staticmethod
#     def check_FSD_ratio(fact, cf, tol=1e-8, direction=+1):
#         """
#         功能: 计算反事实分布相对于事实分布在一阶随机占优 (FSD) 下的满足比例。
#         实现逻辑:
#             - 在所有关键点 (fact ∪ cf) 上分别计算事实与反事实的 ECDF。
#             - 根据方向 (右移优/左移优) 检查是否违反占优条件。
#             - 统计满足条件的点数并计算比例。

#         参数:
#             fact (array-like): 事实分布样本。
#             cf (array-like): 反事实分布样本。
#             tol (float): 数值容差。
#             direction (int): +1 表示反事实应整体右移更优；
#                              -1 表示反事实应整体左移更优。

#         返回:
#             ratio (float): 满足 FSD 条件的比例。
#             total (int): 检查的关键点数量。
#             violations (list): 违反条件的点列表 (x, F_fact, F_cf)。
#         """
#         ecdf_fact = ECDF(fact)
#         ecdf_cf   = ECDF(cf)
#         critical_points = np.unique(np.concatenate([fact, cf]))

#         F_fact = ecdf_fact(critical_points)
#         F_cf   = ecdf_cf(critical_points)

#         if direction == +1:
#             violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Fc > Ff + tol]
#         elif direction == -1:
#             violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Ff > Fc + tol]
#         else:
#             raise ValueError("direction must be +1 (右移优) or -1 (左移优)")

#         satisfied_points = len(critical_points) - len(violations)
#         ratio = satisfied_points / len(critical_points) if len(critical_points) > 0 else 1.0

#         return ratio, len(critical_points), violations

# __all__ = ["FSDConstraintTorch"]



#----------------1001整合check到一个函数---------------------
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
            惩罚系数 λ
        feature_names: list
            特征列名顺序
        dir_map: dict
            方向约束，例如 {"Duration": +1}
        M: int
            采样点数（可选，None 表示全量）
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
        训练阶段的 FSD 惩罚 (可反向传播)
        """
        device = X_fact.device
        total_penalty = torch.zeros(1, device=device, dtype=X_fact.dtype)

        # 🚩 如果设置了 M，对 fact 做采样
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

            # FSD 违规：direction=+1 → cf 应该右移
            violation = torch.clamp(direction * (v_fact - v_cf), min=0.0)

            for mu in mu_list:
                mu = mu.to(device)
                penalty = (mu * violation.pow(2)).sum()
                total_penalty = total_penalty + self.lam * penalty

        return total_penalty

    # ================== 新版统一检查接口 ==================
    @staticmethod
    def check_FSD(fact, cf, tol=1e-8, direction=+1):
        """
        统一的 FSD 检查接口
        返回 dict，包括：
            - holds: 是否严格满足 FSD (bool)
            - violations: 违反点列表
            - point_ratio: 点满足比例
            - interval_ratio: 区间加权满足比例
        """
        ecdf_fact = ECDF(fact)
        ecdf_cf   = ECDF(cf)
        critical_points = np.unique(np.concatenate([fact, cf]))

        F_fact = ecdf_fact(critical_points)
        F_cf   = ecdf_cf(critical_points)

        # 逐点检查
        if direction == +1:
            violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Fc > Ff + tol]
        elif direction == -1:
            violations = [(x, Ff, Fc) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Ff > Fc + tol]
        else:
            raise ValueError("direction must be +1 (右移优) or -1 (左移优)")

        holds = (len(violations) == 0)

        # 点比例
        satisfied_points = len(critical_points) - len(violations)
        point_ratio = satisfied_points / len(critical_points) if len(critical_points) > 0 else 1.0

        # 区间比例（长度加权）
        dx = np.diff(critical_points)
        F_diff = (F_cf - F_fact)[:-1]  # 区间左端点
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

    # ================== 保持画图功能不动 ==================
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

