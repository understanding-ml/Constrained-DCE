# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np  # 仅保留numpy用于绘图，训练逻辑全用PyTorch

# class SSDConstraintTorch(nn.Module):
#     def __init__(self, lam: float = 1.0, feature_names: list = None, dir_map: dict = None,
#                  M: int = None, sample_mode: str = "quantile"):
#         super().__init__()
#         self.lam = lam
#         self.feature_names = feature_names if feature_names is not None else []
#         self.dir_map = dir_map if dir_map is not None else {}
#         self.M = M
#         self.sample_mode = sample_mode

#     # ============ 公共工具函数（纯PyTorch实现，无statsmodels） ============ #
#     @staticmethod
#     def _compute_integrals(fact, cf, sample_M=None, sample_mode="quantile", device=None, for_plot=False):
#         # 确保输入为Tensor（保留梯度）
#         fact_tensor = fact if isinstance(fact, torch.Tensor) else torch.tensor(fact, device=device, dtype=torch.float32)
#         cf_tensor = cf if isinstance(cf, torch.Tensor) else torch.tensor(cf, device=device, dtype=torch.float32)
#         device = fact_tensor.device

#         # 排序（保留梯度）
#         v_fact, _ = torch.sort(fact_tensor)
#         v_cf, _ = torch.sort(cf_tensor)

#         # 纯PyTorch手动实现ECDF（不依赖statsmodels）
#         def torch_ecdf(x, sorted_x):
#             """计算sorted_x处的ECDF值（x为原始数据，sorted_x为排序后的x）"""
#             # 对每个sorted_x中的点，计算x中小于等于它的比例
#             counts = torch.stack([(x <= val).sum() for val in sorted_x])
#             return counts / x.numel()  # 归一化

#         F_fact = torch_ecdf(fact_tensor, v_fact)
#         F_cf = torch_ecdf(cf_tensor, v_cf)

#         # 抽样（训练时用，保持原有逻辑）
#         if sample_M is not None and sample_M < v_fact.numel():
#             if sample_mode == "quantile":
#                 idx_sel = torch.linspace(0, v_fact.numel()-1, sample_M, dtype=torch.long, device=device)
#             elif sample_mode == "random":
#                 idx_sel = torch.randperm(v_fact.numel(), device=device)[:sample_M]
#             v_fact, F_fact = v_fact[idx_sel], F_fact[idx_sel]
#             v_cf, F_cf = v_cf[idx_sel], F_cf[idx_sel]

#         # 矩形法则积分（纯PyTorch实现）
#         def _rectangle_integral(x, F):
#             if x.numel() <= 1:
#                 return torch.tensor([0.0], device=device)
#             dx = x[1:] - x[:-1]
#             integral = torch.cumsum(F[:-1] * dx, dim=0)
#             return torch.cat([torch.tensor([0.0], device=device), integral])

#         A_fact = _rectangle_integral(v_fact, F_fact)
#         A_cf = _rectangle_integral(v_cf, F_cf)

#         # 绘图时转换为numpy（不影响梯度）
#         if for_plot:
#             return (
#                 v_fact.detach().cpu().numpy(),
#                 A_fact.detach().cpu().numpy(),
#                 v_cf.detach().cpu().numpy(),
#                 A_cf.detach().cpu().numpy()
#             )
#         else:
#             return v_fact, A_fact, v_cf, A_cf

#     # ============ forward 训练惩罚（保持不变） ============ #
#     def forward(self, X_fact: torch.Tensor, X_cf: torch.Tensor, mu_list: list) -> torch.Tensor:
#         total_penalty = torch.zeros(1, device=X_fact.device, dtype=X_fact.dtype)

#         for col, direction in self.dir_map.items():
#             if col not in self.feature_names:
#                 continue
#             idx = self.feature_names.index(col)

#             v_fact, v_cf = X_fact[:, idx], X_cf[:, idx]
#             _, A_fact, _, A_cf = self._compute_integrals(
#                 v_fact, v_cf, sample_M=self.M, sample_mode=self.sample_mode, device=X_fact.device
#             )

#             # hinge 罚项（SSD 用积分）
#             S = direction * (A_cf - A_fact)
#             violation = torch.clamp(S, min=0.0)
#             penalty = violation.pow(2).mean()

#             total_penalty = total_penalty + self.lam * penalty

#         return total_penalty

# ============ 1016区间加权 ============ #

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np  # 仅保留numpy用于绘图，训练逻辑全用PyTorch

class SSDConstraintTorch(nn.Module):
    def __init__(self, lam: float = 1.0, feature_names: list = None, dir_map: dict = None,
                 M: int = None, sample_mode: str = "quantile", num_intervals: int = 10):  # ⚠️ 新增区间数量参数
        super().__init__()
        self.lam = lam
        self.feature_names = feature_names if feature_names is not None else []
        self.dir_map = dir_map if dir_map is not None else {}
        self.M = M
        self.sample_mode = sample_mode
        self.num_intervals = num_intervals  # ⚠️ 保存区间数量

    # ============ 公共工具函数（纯PyTorch实现，无statsmodels） ============ #
    @staticmethod
    def _compute_integrals(fact, cf, sample_M=None, sample_mode="quantile", device=None, for_plot=False, num_intervals=10):  # ⚠️ 新增区间参数
        # 确保输入为Tensor（保留梯度）
        fact_tensor = fact if isinstance(fact, torch.Tensor) else torch.tensor(fact, device=device, dtype=torch.float32)
        cf_tensor = cf if isinstance(cf, torch.Tensor) else torch.tensor(cf, device=device, dtype=torch.float32)
        device = fact_tensor.device

        # 排序（保留梯度）
        v_fact, _ = torch.sort(fact_tensor)
        v_cf, _ = torch.sort(cf_tensor)

        # ⚠️ 新增：计算区间权重（仅针对事实分布）
        def compute_interval_weights(sorted_data, num_intervals):
            if sorted_data.numel() <= 1:
                return torch.ones_like(sorted_data, device=device)  # 单个样本时权重为1
            # 划分区间边界（基于事实分布的最小值和最大值）
            min_val = sorted_data.min()
            max_val = sorted_data.max()
            intervals = torch.linspace(min_val, max_val, num_intervals + 1, device=device)  # 区间边界
            # 计算每个样本所属区间
            interval_indices = torch.bucketize(sorted_data, intervals[1:-1])  # 0~num_intervals-1
            # 统计每个区间的样本数
            counts = torch.bincount(interval_indices, minlength=num_intervals)
            # 计算每个样本的权重（区间总权重=1/num_intervals，再均分）
            weights = 1.0 / (counts[interval_indices] * num_intervals)
            return weights

        # 计算事实分布样本的区间权重
        fact_weights = compute_interval_weights(v_fact, num_intervals)  # ⚠️ 区间权重

        # 纯PyTorch手动实现ECDF（不依赖statsmodels）
        def torch_ecdf(x, sorted_x):
            """计算sorted_x处的ECDF值（x为原始数据，sorted_x为排序后的x）"""
            counts = torch.stack([(x <= val).sum() for val in sorted_x])
            return counts / x.numel()  # 归一化

        F_fact = torch_ecdf(fact_tensor, v_fact)
        F_cf = torch_ecdf(cf_tensor, v_cf)

        # 抽样（训练时用，保持原有逻辑）
        if sample_M is not None and sample_M < v_fact.numel():
            if sample_mode == "quantile":
                idx_sel = torch.linspace(0, v_fact.numel()-1, sample_M, dtype=torch.long, device=device)
            elif sample_mode == "random":
                idx_sel = torch.randperm(v_fact.numel(), device=device)[:sample_M]
            v_fact, F_fact = v_fact[idx_sel], F_fact[idx_sel]
            v_cf, F_cf = v_cf[idx_sel], F_cf[idx_sel]
            fact_weights = fact_weights[idx_sel]  # ⚠️ 抽样时同步调整权重

        # 矩形法则积分（纯PyTorch实现）
        def _rectangle_integral(x, F):
            if x.numel() <= 1:
                return torch.tensor([0.0], device=device)
            dx = x[1:] - x[:-1]
            integral = torch.cumsum(F[:-1] * dx, dim=0)
            return torch.cat([torch.tensor([0.0], device=device), integral])

        A_fact = _rectangle_integral(v_fact, F_fact)
        A_cf = _rectangle_integral(v_cf, F_cf)

        # 绘图时转换为numpy（不影响梯度）
        if for_plot:
            return (
                v_fact.detach().cpu().numpy(),
                A_fact.detach().cpu().numpy(),
                v_cf.detach().cpu().numpy(),
                A_cf.detach().cpu().numpy()
            )
        else:
            return v_fact, A_fact, v_cf, A_cf, fact_weights  # ⚠️ 新增返回权重

    # ============ forward 训练惩罚（修改为加权惩罚） ============ #
    def forward(self, X_fact: torch.Tensor, X_cf: torch.Tensor, mu_list: list) -> torch.Tensor:
        total_penalty = torch.zeros(1, device=X_fact.device, dtype=X_fact.dtype)

        for col, direction in self.dir_map.items():
            if col not in self.feature_names:
                continue
            idx = self.feature_names.index(col)

            v_fact, v_cf = X_fact[:, idx], X_cf[:, idx]
            # ⚠️ 接收新增的权重返回值
            _, A_fact, _, A_cf, fact_weights = self._compute_integrals(
                v_fact, v_cf, sample_M=self.M, sample_mode=self.sample_mode, 
                device=X_fact.device, num_intervals=self.num_intervals  # ⚠️ 传入区间数量
            )

            # hinge 罚项（SSD 用积分）
            S = direction * (A_cf - A_fact)
            violation = torch.clamp(S, min=0.0)
            # ⚠️ 修改为加权平均（用事实分布的区间权重）
            penalty = (violation.pow(2) * fact_weights).sum() / fact_weights.sum()

            total_penalty = total_penalty + self.lam * penalty

        return total_penalty
    # ============ 统一的检查函数（保持不变） ============ #
    @staticmethod
    def check_SSD(fact, cf, tol=1e-8, return_violations=False, direction=+1):
        fact = torch.tensor(fact, dtype=torch.float32) if not isinstance(fact, torch.Tensor) else fact
        cf = torch.tensor(cf, dtype=torch.float32) if not isinstance(cf, torch.Tensor) else cf

        critical_points = torch.sort(fact)[0]
        F_fact = torch.arange(1, len(critical_points) + 1, dtype=torch.float32) / len(critical_points)
        F_cf = torch.tensor([(cf <= x).sum() for x in critical_points], dtype=torch.float32) / len(cf)

        if direction == +1:
            violations = [(x.item(), Ff.item(), Fc.item()) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Fc > Ff + tol]
        elif direction == -1:
            violations = [(x.item(), Ff.item(), Fc.item()) for x, Ff, Fc in zip(critical_points, F_fact, F_cf) if Ff > Fc + tol]
        else:
            raise ValueError("direction must be +1 (右移优) or -1 (左移优)")

        holds = (len(violations) == 0)
        satisfied_points = len(critical_points) - len(violations)
        point_ratio = satisfied_points / len(critical_points) if len(critical_points) > 0 else 1.0

        dx = torch.diff(critical_points)
        F_diff = (F_cf - F_fact)[:-1]
        if direction == +1:
            ok_intervals = (F_diff <= tol)
        else:
            ok_intervals = (F_diff >= -tol)
        interval_ratio = dx[ok_intervals].sum().item() / dx.sum().item() if dx.sum().item() > 0 else 1.0

        return {
            "holds": holds,
            "violations": violations,
            "point_ratio": point_ratio,
            "interval_ratio": interval_ratio
        }

    # ============ 可视化函数（保持不变） ============ #
    # @staticmethod
    # def plot_cdf_with_integral(fact, cf, feature_name="", direction=+1):
    #     v_fact, A_fact, v_cf, A_cf = SSDConstraintTorch._compute_integrals(
    #         fact, cf, sample_M=None, for_plot=True
    #     )

    #     F_fact = np.arange(1, len(v_fact) + 1, dtype=np.float32) / len(v_fact)
    #     F_cf = np.arange(1, len(v_cf) + 1, dtype=np.float32) / len(v_cf)

    #     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    #     axes[0].step(v_fact, F_fact, label=f"Factual {feature_name}", color="blue", where='post')
    #     axes[0].step(v_cf, F_cf, label=f"Counterfactual {feature_name}", color="orange", where='post')
    #     axes[0].set_title(f"CDF Comparison ({'Right shift' if direction==+1 else 'Left shift'} better)")
    #     axes[0].set_xlabel(feature_name)
    #     axes[0].set_ylabel("CDF")
    #     axes[0].legend()
    #     axes[0].grid(True, linestyle="--", alpha=0.7)

    #     axes[1].plot(v_fact, A_fact, label=f"Factual Integral {feature_name}", color="blue")
    #     axes[1].plot(v_cf, A_cf, label=f"Counterfactual Integral {feature_name}", color="orange")
    #     axes[1].set_title("Integral CDF (SSD check)")
    #     axes[1].set_xlabel(feature_name)
    #     axes[1].set_ylabel("∫CDF dx")
    #     axes[1].legend()
    #     axes[1].grid(True, linestyle="--", alpha=0.7)

    #     plt.tight_layout()
    #     plt.show()
# ============ 可视化函数（只修改此部分） ============ #
# ============ 可视化函数（只修改此部分） ============ #
    @staticmethod
    def plot_cdf_with_integral(fact, cf, feature_name="", direction=+1):
        """
        并排绘制 CDF 和积分 CDF（手动计算CDF，而不是从积分中取）
        """
        # 调用_compute_integrals，使用全量数据（sample_M=None）并获取numpy格式结果
        v_fact, A_fact, v_cf, A_cf = SSDConstraintTorch._compute_integrals(
            fact, cf, 
            sample_M=None,  # 绘图用全量数据，不抽样
            for_plot=True   # 返回numpy格式，适配绘图
        )

        # 手动计算 CDF
        def compute_cdf(data):
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            return sorted_data, cdf

        v_fact_cdf, F_fact = compute_cdf(v_fact)
        v_cf_cdf, F_cf = compute_cdf(v_cf)

        # 计算违反点
        violations = []
        for i, (Af, Ac) in enumerate(zip(A_fact, A_cf)):  # 只用A_fact 和 A_cf
            if (Ac > Af + 1e-8 and direction == 1) or (Af > Ac + 1e-8 and direction == -1):
                violations.append((v_fact[i], Af, Ac))  # 保存违反点，x用v_fact来标识

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：CDF对比（绘制的是Factual 和 Counterfactual的 CDF）
        axes[0].step(v_fact_cdf, F_fact, label=f"Factual {feature_name}", color="blue", where='post')
        axes[0].step(v_cf_cdf, F_cf, label=f"Counterfactual {feature_name}", color="orange", where='post')
        axes[0].set_title(f"CDF Comparison ({'Right shift' if direction==+1 else 'Left shift'} better)")
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel("CDF")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.7)

        # 右图：积分CDF对比（绘制的是Factual Integral 和 Counterfactual Integral）
        axes[1].plot(v_fact, A_fact, label=f"Factual Integral {feature_name}", color="blue")
        axes[1].plot(v_cf, A_cf, label=f"Counterfactual Integral {feature_name}", color="orange")

        # 标注违反点
        if violations:
            violation_x = [v[0] for v in violations]  # 从 violations 中取 x 坐标
            violation_y = [v[1] for v in violations]  # 从 violations 中取 A_fact 作为 y 坐标
            axes[1].scatter(violation_x, violation_y, color="red", marker="x", label="Violations")

        axes[1].set_title("Integral CDF (SSD check)")
        axes[1].set_xlabel(feature_name)
        axes[1].set_ylabel("∫CDF dx")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()




__all__ = ["SSDConstraintTorch"]
    










 