# import torch

# class Evaluator:
#     def __init__(self, mode="threshold", theta=0.5, tau=0.2):
#         """
#         mode: "threshold" or "distance"
#         theta: 阈值分类模式下的分界点 (默认0.5)
#         tau: 距离模式下的误差容忍度 (默认0.2)
#         """
#         assert mode in ["threshold", "distance"], "mode must be 'threshold' or 'distance'"
#         self.mode = mode
#         self.theta = theta
#         self.tau = tau

#     def evaluate(self, y_target: torch.Tensor, y_cf: torch.Tensor):
#         """
#         输入:
#           y_target: [N] torch.Tensor, 可以是0/1或概率
#           y_cf:     [N] torch.Tensor, 模型输出（0~1 概率）
#         输出:
#           R_mask: [N] torch.BoolTensor, True表示失败(需要回收)
#           G_mask: [N] torch.BoolTensor, True表示成功(不用回收)
#         """
#         if self.mode == "threshold":
#             # 先把概率转为 hard label
#             target_label = (y_target >= self.theta).long()
#             outcome_label = (y_cf >= self.theta).long()
#             R_mask = (target_label != outcome_label)
#         else:  # distance 模式
#             delta = torch.abs(y_cf - y_target)
#             R_mask = (delta > self.tau)

#         G_mask = ~R_mask
#         return R_mask, G_mask
    

# import torch
# from explainers.distances import WassersteinDivergence   # 引入瓦瑟斯坦距离计算类
# import ot  # 导入POT库


# class Evaluator:
#     def __init__(self, mode="threshold", theta=0.5, tau=0.2, reg=1):
#         """
#         mode: "threshold" or "distance"
#         theta: 阈值分类模式下的分界点 (默认0.5)
#         tau: 距离模式下的误差容忍度 (默认0.2)
#         reg: 瓦瑟斯坦距离的正则化参数 (默认1)
#         """
#         assert mode in ["threshold", "distance"], "mode must be 'threshold' or 'distance'"
#         self.mode = mode
#         self.theta = theta
#         self.tau = tau
#         self.reg = reg  # 新增：保存正则化参数为实例属性
        
#         # 实例化瓦瑟斯坦距离计算器
#         self.wasserstein = WassersteinDivergence(reg=reg)

#     def evaluate(self, y_target: torch.Tensor, y_cf: torch.Tensor, wasserstein_matrix=None):
#         """
#         输入:
#           y_target: [N] torch.Tensor, 可以是0/1或概率
#           y_cf:     [N] torch.Tensor, 模型输出（0~1 概率）
#           wasserstein_matrix: [N, N] torch.Tensor, 计算的瓦瑟斯坦距离矩阵
#         输出:
#           R_mask: [N] torch.BoolTensor, True表示失败(需要回收)
#           G_mask: [N] torch.BoolTensor, True表示成功(不用回收)
#         """
#         if self.mode == "threshold":
#             # 先把概率转为 hard label
#             target_label = (y_target >= self.theta).long()
#             outcome_label = (y_cf >= self.theta).long()
#             R_mask = (target_label != outcome_label)
#         elif self.mode == "distance":  # distance 模式
#         #     # 计算每个 y_cf 和对应的最大传输权重的 y_target
#         #     R_mask = torch.zeros_like(y_target, dtype=torch.bool)
            
#         #     # 瓦瑟斯坦距离计算（每个 y_cf 和 y_target）
#         #     dist, nu = self.wasserstein.distance(y_cf, y_target, delta=0.05)  # 计算瓦瑟斯坦距离矩阵
            
#         #     # 打印瓦瑟斯坦距离矩阵
#         #     print(f"瓦瑟斯坦距离矩阵 nu: {nu}")

#         #     for i in range(len(y_cf)):  # 对于每个 y_cf
#         #         max_weight_idx = torch.argmax(nu[i])  # 找到最大传输权重的 y_target
#         #         selected_y_target = y_target[max_weight_idx]  # 获取该最大权重对应的 y_target
                
#         #         # 计算 y_cf 和 selected_y_target 的差异
#         #         diff = torch.abs(y_cf[i] - selected_y_target)

#         #         # 打印每个 y_cf 和对应的最大传输权重的 y_target 及差异
#         #         print(f"y_cf[{i}] = {y_cf[i]:.4f}, selected_y_target = {selected_y_target:.4f}, diff = {diff:.4f}")

#         #         if diff > self.tau:  # 如果差异大于 tau，标记为失败 (需要回收)
#         #             R_mask[i] = True

#         # G_mask = ~R_mask  # 成功的标记为 G_mask
#         # return R_mask, G_mask

#               R_mask = torch.zeros_like(y_target, dtype=torch.bool)
            
#               # 将输入转换为numpy数组（POT库通常处理numpy数组）
#               y_cf_np = y_cf.cpu().detach().numpy().reshape(-1, 1)  # 转换为[N,1]形状
#               y_target_np = y_target.cpu().detach().numpy().reshape(-1, 1)  # 转换为[N,1]形状
              
#               # 计算代价矩阵（使用L2距离）
#               M = ot.dist(y_cf_np, y_target_np, metric='euclidean')  # 形状为[N, N]
              
#               # 定义边际分布（均匀分布）
#               a = ot.unif(y_cf_np.shape[0])  # y_cf侧的边际分布
#               b = ot.unif(y_target_np.shape[0])  # y_target侧的边际分布
              
#               # 计算最优传输矩阵（使用正则化OT，与原代码的正则化参数reg对应）
#               nu = ot.sinkhorn(a, b, M, reg=self.reg)  # 形状为[N, N]，即传输矩阵
              
#               # 转换回torch张量（保持与原代码数据类型一致）
#               nu = torch.from_numpy(nu).to(y_cf.device)  # 移回与输入相同的设备
              
#               # 打印瓦瑟斯坦距离矩阵
#               print(f"瓦瑟斯坦距离矩阵 nu: {nu}")

#               for i in range(len(y_cf)):  # 对于每个 y_cf
#                   max_weight_idx = torch.argmax(nu[i])  # 找到最大传输权重的 y_target索引
#                   selected_y_target = y_target[max_weight_idx]  # 获取对应y_target
                  
#                   # 计算差异
#                   diff = torch.abs(y_cf[i] - selected_y_target)

#                   # 打印详细信息
#                   print(f"y_cf[{i}] = {y_cf[i]:.4f}, selected_y_target = {selected_y_target:.4f}, diff = {diff:.4f}")

#                   if diff > self.tau:  # 超过阈值标记为失败
#                       R_mask[i] = True

#         G_mask = ~R_mask  # 成功的标记为 G_mask
#         return R_mask, G_mask


import torch
from explainers.distances import WassersteinDivergence
import ot
import numpy as np


class Evaluator:
    def __init__(self, mode="threshold", theta=0.5, tau=0.1, reg=1):
        assert mode in ["threshold", "distance"], "mode must be 'threshold' or 'distance'"
        self.mode = mode
        self.theta = theta
        self.tau = tau
        self.reg = reg  # 保持原reg参数，不强制减小
        self.wasserstein = WassersteinDivergence(reg=reg)

    def evaluate(self, y_target: torch.Tensor, y_cf: torch.Tensor):
        if self.mode == "threshold":
            target_label = (y_target >= self.theta).long()
            outcome_label = (y_cf >= self.theta).long()
            R_mask = (target_label != outcome_label)
        elif self.mode == "distance":
            R_mask = torch.zeros_like(y_cf, dtype=torch.bool)
            N = len(y_cf)
            M = len(y_target)  # 允许y_cf和y_target长度不同
            
            # 转换为numpy数组（形状[N,1]和[M,1]）
            y_cf_np = y_cf.cpu().detach().numpy().reshape(-1, 1)
            y_target_np = y_target.cpu().detach().numpy().reshape(-1, 1)
            
            # 计算全局代价矩阵（N×M，非单个样本）
            # 用欧氏距离，不刻意放大，保持原始尺度
            M_cost = ot.dist(y_cf_np, y_target_np, metric='euclidean')
            
            # 非one-hot边际分布：均匀分布（总质量为1，每行和为1/N）
            a = ot.unif(N)  # [N]，每个元素为1/N，总和为1
            b = ot.unif(M)  # [M]，每个元素为1/M，总和为1
            
            # 计算最优传输矩阵（N×M）
            nu = ot.sinkhorn(a, b, M_cost, reg=self.reg, numItermax=100000)
            
            # 转换回torch张量
            nu = torch.from_numpy(nu).to(y_cf.device)
            
            # # 打印行和（应接近1/N）和每行最大权重
            # print(f"传输矩阵行和（应接近1/{N}）: {nu.sum(dim=1)}")
            # print(f"每行最大权重: {nu.max(dim=1).values}")

            for i in range(N):
                # 找每行最大权重对应的y_target索引
                max_weight_idx = torch.argmax(nu[i])
                max_weight_idx = torch.clamp(max_weight_idx, 0, M-1)  # 索引安全
                
                selected_y_target = y_target[max_weight_idx]
                diff = torch.abs(y_cf[i] - selected_y_target)
                
                # print(f"y_cf[{i}] = {y_cf[i]:.4f}, 匹配目标 = {selected_y_target:.4f}, 差异 = {diff:.4f}")
                
                if diff > self.tau:
                    R_mask[i] = True

        G_mask = ~R_mask
        return R_mask, G_mask