import torch

class MeanRecovery:
    def __init__(self, feature_names, bounds: dict, max_drop_ratio=1, verbose=False):
        """
        Mean Recovery Algorithm: Adjust the mean of specific features to fall within given intervals by removing sample points in the R set
        """
        self.feature_names = feature_names
        self.name2idx = {n: i for i, n in enumerate(feature_names)}  # Column name → index
        self.bounds = bounds
        self.max_drop_ratio = max_drop_ratio
        self.verbose = verbose

        # New: Record the "original row indices" deleted in this recover operation
        self.last_dropped_indices = []
        # (Optional) Record by feature
        self.last_dropped_indices_by_feature = {}

    def recover(self, X_cf, R_mask):
        """
        Adjust the mean by removing extreme points based on the R set indicated by R_mask
        Returns the repaired sample matrix (some R samples may be removed)
        """
        X = X_cf.clone()
        N0 = X.shape[0]  # Initial sample count for max_drop_ratio calculation
        # Maintain mapping from "current view index" to "original global index"
        alive_idx = torch.arange(N0, device=X.device)

        # Clear records before each call
        self.last_dropped_indices = []
        self.last_dropped_indices_by_feature = {}

        # Iterate over all features subject to mean constraints
        for col, (low, high) in self.bounds.items():
            j = self.name2idx[col]
            xj = X[:, j]
            mu = xj.mean().item()

            if low <= mu <= high:
                if self.verbose:
                    print(f"[MeanRecovery] {col} success: mean={mu:.3f} ∈ [{low}, {high}]")
                continue

            # Mean too small → remove small values; Mean too large → remove large values
            order = torch.argsort(xj, descending=(mu > high))

            keep = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
            dropped = 0
            success = False

            # Record original indices deleted for this feature
            dropped_this_feature = []

            for idx in order:
                # Only allow deletion of points in R set: map to original index for judgment
                orig_idx = alive_idx[idx].item()
                if not R_mask[orig_idx]:
                    continue

                keep[idx] = False
                dropped += 1
                # Record the "original index" of deleted sample
                dropped_this_feature.append(orig_idx)
                self.last_dropped_indices.append(orig_idx)

                mu_now = X[keep, j].mean().item()

                # Stop and commit deletion once mean enters target interval
                if low <= mu_now <= high:
                    X = X[keep]
                    alive_idx = alive_idx[keep]   
                    success = True
                    if self.verbose:
                        print(f"[MeanRecovery] {col} success: mean={mu_now:.3f} ∈ [{low}, {high}] (dropped {dropped})")
                    break

                # Exceed maximum drop ratio (relative to initial N0)
                if dropped >= int(self.max_drop_ratio * N0):
                    mu_now = X[keep, j].mean().item()
                    if self.verbose:
                        print(f"[MeanRecovery] {col} partial fix: mean={mu_now:.3f} ∉ [{low}, {high}], "
                              f"max drop ratio reached (dropped {dropped})")
                    X = X[keep]
                    alive_idx = alive_idx[keep]   
                    break

            # Save drop records for this feature
            self.last_dropped_indices_by_feature[col] = dropped_this_feature

            # Commit current keep if loop ends without success (candidate points exhausted)
            if not success and keep.sum().item() < X.shape[0]:
                X = X[keep]
                alive_idx = alive_idx[keep]     

        return X
