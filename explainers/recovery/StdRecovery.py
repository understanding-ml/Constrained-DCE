import torch

class StdRecovery:
    def __init__(self, feature_names, bounds: dict, max_drop_ratio=0.5, verbose=False):
        """
        Standard Deviation Recovery Algorithm: Reduce the standard deviation of specific features to below the given upper bound by removing sample points in the R set
        """
        self.feature_names = feature_names
        self.name2idx = {n: i for i, n in enumerate(feature_names)}
        self.bounds = bounds
        self.max_drop_ratio = max_drop_ratio
        self.verbose = verbose

        # New: Record the original row indices of deleted samples
        self.last_dropped_indices = []
        self.last_dropped_indices_by_feature = {}

    def recover(self, X_cf, R_mask):
        """
        Recovery Function: Reduce standard deviation by removing extreme points based on the R set indicated by R_mask
        """
        X = X_cf.clone()
        N0 = X.shape[0]  # Initial sample count for ratio calculation
        alive_idx = torch.arange(N0, device=X.device)  #  Original index mapping

        current_R_mask = R_mask.clone()  # Initialize current_R_mask with the same length as input R_mask

        # Clear records before each call
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
            dropped_this_feature = []  # Drop records for this feature

            while True:
                mu = X[keep, j].mean()
                candidates = torch.nonzero(current_R_mask & keep, as_tuple=True)[0]

                if len(candidates) == 0:
                    std_now = X[keep, j].std(unbiased=False).item()
                    if self.verbose:
                        print(f"[StdRecovery] {col} partial fix: std={std_now:.3f} > {high}, "
                              f"no more candidates to drop")
                    X = X[keep]
                    alive_idx = alive_idx[keep]
                    break

                # Split into left and right groups
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
                    current_R_mask = current_R_mask[keep]  
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
                    current_R_mask = current_R_mask[keep]  
                    break

            # Save drop records for this feature
            self.last_dropped_indices.extend(dropped_this_feature)
            self.last_dropped_indices_by_feature[col] = dropped_this_feature

            if not success:
                X = X[keep]
                alive_idx = alive_idx[keep]
                current_R_mask = current_R_mask[keep]  

        return X
