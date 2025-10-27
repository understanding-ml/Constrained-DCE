import torch
from explainers.recovery.MeanRecovery import MeanRecovery
from explainers.recovery.StdRecovery import StdRecovery


class RecoveryManager:
    def __init__(self, feature_names, configs, verbose=False):
        """
        RecoveryManager: 带优先级调度的回收管理器

        参数
        ----
        feature_names: list
            特征名顺序
        configs: list of dict
            每个约束的配置，例如:
            [
                {"type": "mean", "bounds": {"Duration": (0.3, 0.6)}, "priority": 1},
                {"type": "std",  "bounds": {"Duration": 0.4},        "priority": 2},
                {"type": "mean", "bounds": {"Credit": (0.2, 0.5)},  "priority": 0}
            ]
        verbose: bool
            是否打印调试信息
        """
        self.feature_names = feature_names
        self.configs = configs
        self.verbose = verbose

        # 将 recovery 算法实例化，并挂上 priority
        self.recoveries = []
        for cfg in configs:
            ctype = cfg["type"].lower()
            priority = cfg.get("priority", 999)  # 未指定时排最后
            if ctype == "mean":
                rec = MeanRecovery(feature_names, cfg["bounds"], verbose=verbose)
                self.recoveries.append((priority, "mean", rec))
            elif ctype == "std":
                rec = StdRecovery(feature_names, cfg["bounds"], verbose=verbose)
                self.recoveries.append((priority, "std", rec))
            else:
                raise ValueError(f"Unsupported recovery type: {ctype}")

        # 按 priority 排序
        self.recoveries.sort(key=lambda x: x[0])

    def run(self, X_cf, R_mask):
        """
        按照 priority 执行 recovery 算法
        """
        X = X_cf.clone()
        mask = R_mask.clone()
        report = {}

        # 先按 priority 排好 → 再按 feature 分组（同一 feature 内保证 mean→std）
        feature_groups = {}
        for priority, ctype, rec in self.recoveries:
            for feat in rec.bounds.keys():
                feature_groups.setdefault(priority, {}).setdefault(feat, []).append((ctype, rec))

        # 遍历 priority 层次
        for priority in sorted(feature_groups.keys()):
            for feat, recovs in feature_groups[priority].items():
                report.setdefault(feat, {})

                # 保证同一维度 mean 在前，std 在后
                recovs.sort(key=lambda x: 0 if x[0] == "mean" else 1)

                for ctype, recovery in recovs:
                    # 修复前检查是否已满足
                    if ctype == "mean":
                        low, high = recovery.bounds[feat]
                        mu = X[:, recovery.name2idx[feat]].mean().item()
                        if low <= mu <= high:
                            report[feat]["mean"] = "ok"
                            continue
                    elif ctype == "std":
                        high = recovery.bounds[feat]
                        std = X[:, recovery.name2idx[feat]].std(unbiased=False).item()
                        if std <= high:
                            report[feat]["std"] = "ok"
                            continue

                    # 调用具体 recovery
                    X_new = recovery.recover(X, mask)

                    # 判断结果
                    if ctype == "mean":
                        mu = X_new[:, recovery.name2idx[feat]].mean().item()
                        low, high = recovery.bounds[feat]
                        report[feat]["mean"] = "success" if low <= mu <= high else "partial"
                    elif ctype == "std":
                        std = X_new[:, recovery.name2idx[feat]].std(unbiased=False).item()
                        high = recovery.bounds[feat]
                        report[feat]["std"] = "success" if std <= high else "partial"

                    # 更新 X 与 mask
                    if X_new.shape[0] != X.shape[0]:
                        kept_idx = torch.arange(X.shape[0])[mask][:X_new.shape[0]]
                        mask = mask[kept_idx]
                    X = X_new

        return X, report
