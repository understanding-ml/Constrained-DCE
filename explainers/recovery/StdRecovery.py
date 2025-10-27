import torch

class StdRecovery:
    def __init__(self, feature_names, bounds: dict, max_drop_ratio=0.5, verbose=False):
        """
        标准差回收算法：通过删除 R 集合中的样本点，尽量将某些特征的标准差压到给定上界以下
        """
        self.feature_names = feature_names
        self.name2idx = {n: i for i, n in enumerate(feature_names)}
        self.bounds = bounds
        self.max_drop_ratio = max_drop_ratio
        self.verbose = verbose

        # ✅ 新增：记录删除的原始行索引
        self.last_dropped_indices = []
        self.last_dropped_indices_by_feature = {}

    def recover(self, X_cf, R_mask):
        """
        回收函数：根据 R_mask 指示的 R 集合，删除极端点以降低标准差
        """
        X = X_cf.clone()
        N0 = X.shape[0]  # 初始样本数用于比例
        alive_idx = torch.arange(N0, device=X.device)  # ✅ 原始索引映射

        # 必须在函数开头初始化！这行漏了就会报UnboundLocalError
        current_R_mask = R_mask.clone()  # 初始化当前R_mask，与输入R_mask长度一致

        # 每次调用先清空
        self.last_dropped_indices = []
        self.last_dropped_indices_by_feature = {}

        for col, high in self.bounds.items():
            j = self.name2idx[col]
            std = X[:, j].std(unbiased=False).item()

            if std <= high:
                if self.verbose:
                    print(f"[StdRecovery] {col} success: std={std:.3f} ≤ {high}")
                continue

            keep = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
            dropped = 0
            success = False
            dropped_this_feature = []  # ✅ 本特征的 drop 记录

            while True:
                mu = X[keep, j].mean()
                # candidates = torch.nonzero(R_mask[alive_idx] & keep, as_tuple=True)[0]
                # 唯一关键修改：用 current_R_mask 替代 R_mask[alive_idx]
                candidates = torch.nonzero(current_R_mask & keep, as_tuple=True)[0]

                if len(candidates) == 0:
                    std_now = X[keep, j].std(unbiased=False).item()
                    if self.verbose:
                        print(f"[StdRecovery] {col} partial fix: std={std_now:.3f} > {high}, "
                              f"no more candidates to drop")
                    X = X[keep]
                    alive_idx = alive_idx[keep]
                    break

                # 左右分组
                left = [idx.item() for idx in candidates if X[idx, j] < mu]
                right = [idx.item() for idx in candidates if X[idx, j] > mu]

                if len(left) > 0 and len(right) > 0:
                    left_idx = max(left, key=lambda idx: abs(X[idx, j] - mu).item())
                    right_idx = max(right, key=lambda idx: abs(X[idx, j] - mu).item())
                    keep[left_idx] = False
                    keep[right_idx] = False
                    dropped += 2
                    dropped_this_feature.extend([alive_idx[left_idx].item(),
                                                 alive_idx[right_idx].item()])
                else:
                    idx = max(candidates, key=lambda idx: abs(X[idx, j] - mu).item())
                    keep[idx] = False
                    dropped += 1
                    dropped_this_feature.append(alive_idx[idx].item())

                std_now = X[keep, j].std(unbiased=False).item()

                if std_now <= high:
                    X = X[keep]
                    alive_idx = alive_idx[keep]
                    success = True
                    current_R_mask = current_R_mask[keep]  # ✅ 同步裁剪 R_mask
                    if self.verbose:
                        print(f"[StdRecovery] {col} success: std={std_now:.3f} ≤ {high} "
                              f"(dropped {dropped})")
                    break

                if dropped >= int(self.max_drop_ratio * N0):
                    if self.verbose:
                        print(f"[StdRecovery] {col} partial fix: std={std_now:.3f} > {high}, "
                              f"max drop ratio reached (dropped {dropped})")
                    X = X[keep]
                    alive_idx = alive_idx[keep]
                    current_R_mask = current_R_mask[keep]  # ✅ 同步裁剪 R_mask
                    break

            # 保存本特征的 drop 记录
            self.last_dropped_indices.extend(dropped_this_feature)
            self.last_dropped_indices_by_feature[col] = dropped_this_feature

            if not success:
                X = X[keep]
                alive_idx = alive_idx[keep]
                current_R_mask = current_R_mask[keep]  # ✅ 同步裁剪 R_mask

        return X
