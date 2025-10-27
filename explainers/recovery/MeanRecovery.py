import torch

class MeanRecovery:
    def __init__(self, feature_names, bounds: dict, max_drop_ratio=1, verbose=False):
        """
        均值回收算法：通过删除 R 集合中的样本点，尽量将某些特征的均值调整到给定区间内
        """
        self.feature_names = feature_names
        self.name2idx = {n: i for i, n in enumerate(feature_names)}  # 列名 → 索引
        self.bounds = bounds
        self.max_drop_ratio = max_drop_ratio
        self.verbose = verbose

        # ✅ 新增：记录本次 recover 被删除的“原始行索引”
        self.last_dropped_indices = []
        # （可选）按特征记录
        self.last_dropped_indices_by_feature = {}

    def recover(self, X_cf, R_mask):
        """
        根据 R_mask 指示的 R 集合，删除极端点以调整均值
        返回修复后的样本矩阵（可能删除了一部分 R 样本）
        """
        X = X_cf.clone()
        N0 = X.shape[0]  # 初始样本数用于 max_drop_ratio
        # ✅ 维护从“当前视图索引”到“原始全局索引”的映射
        alive_idx = torch.arange(N0, device=X.device)

        # ✅ 每次调用先清空记录
        self.last_dropped_indices = []
        self.last_dropped_indices_by_feature = {}

        # 遍历所有需要进行均值约束的特征
        for col, (low, high) in self.bounds.items():
            j = self.name2idx[col]
            xj = X[:, j]
            mu = xj.mean().item()

            if low <= mu <= high:
                if self.verbose:
                    print(f"[MeanRecovery] {col} success: mean={mu:.3f} ∈ [{low}, {high}]")
                continue

            # 均值偏小 → 删小值；均值偏大 → 删大值
            order = torch.argsort(xj, descending=(mu > high))

            keep = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
            dropped = 0
            success = False

            # 记录该特征被删的原始索引
            dropped_this_feature = []

            for idx in order:
                # 只允许删除 R 集内的点：需要把“原始索引”映射出来判断
                orig_idx = alive_idx[idx].item()
                if not R_mask[orig_idx]:
                    continue

                keep[idx] = False
                dropped += 1
                # ✅ 记录被删的“原始索引”
                dropped_this_feature.append(orig_idx)
                self.last_dropped_indices.append(orig_idx)

                mu_now = X[keep, j].mean().item()

                # 均值进入区间则立即提交删除并停止
                if low <= mu_now <= high:
                    X = X[keep]
                    alive_idx = alive_idx[keep]   # ✅ 同步映射
                    success = True
                    if self.verbose:
                        print(f"[MeanRecovery] {col} success: mean={mu_now:.3f} ∈ [{low}, {high}] (dropped {dropped})")
                    break

                # 超过最大删除比例（相对于初始 N0）
                if dropped >= int(self.max_drop_ratio * N0):
                    mu_now = X[keep, j].mean().item()
                    if self.verbose:
                        print(f"[MeanRecovery] {col} partial fix: mean={mu_now:.3f} ∉ [{low}, {high}], "
                              f"max drop ratio reached (dropped {dropped})")
                    X = X[keep]
                    alive_idx = alive_idx[keep]   # ✅ 同步映射
                    break

            # 把本特征的 drop 记录保存下来
            self.last_dropped_indices_by_feature[col] = dropped_this_feature

            # 若循环走完仍未成功（候选点用尽），也要提交当前 keep
            if not success and keep.sum().item() < X.shape[0]:
                X = X[keep]
                alive_idx = alive_idx[keep]       # ✅ 同步映射

        return X
