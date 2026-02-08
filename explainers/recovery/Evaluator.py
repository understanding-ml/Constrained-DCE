
import torch
from explainers.distances import WassersteinDivergence
import ot
import numpy as np


class Evaluator:
    '''
    Core Functionality:
    This class is a quality evaluator for counterfactual generation tasks, which judges whether generated counterfactual samples (y_cf) meet the target requirements (y_target) through two modes, and outputs boolean masks to mark samples that need to be rejected/retained.
    
    Key Logic:
    1. Threshold mode: Convert y_target and y_cf to hard labels (0/1) based on theta, mark samples with inconsistent labels as "failed (to be rejected)".
    2. Distance mode: Use Optimal Transport (OT) to match each y_cf with the most relevant y_target, calculate the absolute difference between them; mark samples with difference exceeding tau as "failed (to be rejected)".
    
    Output:
    - R_mask (Reject mask): Boolean tensor where True means the sample fails and needs to be rejected/discarded.
    - G_mask (Good mask): Boolean tensor where True means the sample succeeds and can be retained (inverse of R_mask).
    '''
    def __init__(self, mode="threshold", theta=0.5, tau=0.1, reg=1):
        assert mode in ["threshold", "distance"], "mode must be 'threshold' or 'distance'"
        self.mode = mode
        self.theta = theta
        self.tau = tau
        self.reg = reg  # Keep the original reg parameter without forced reduction
        self.wasserstein = WassersteinDivergence(reg=reg)

    def evaluate(self, y_target: torch.Tensor, y_cf: torch.Tensor):
        if self.mode == "threshold":
            target_label = (y_target >= self.theta).long()
            outcome_label = (y_cf >= self.theta).long()
            R_mask = (target_label != outcome_label)
        elif self.mode == "distance":
            R_mask = torch.zeros_like(y_cf, dtype=torch.bool)
            N = len(y_cf)
            M = len(y_target)  # Allow different lengths for y_cf and y_target
            
            # Convert to numpy arrays (shape [N,1] and [M,1])
            y_cf_np = y_cf.cpu().detach().numpy().reshape(-1, 1)
            y_target_np = y_target.cpu().detach().numpy().reshape(-1, 1)
            
            # Calculate global cost matrix (N×M, not per-sample)
            # Use Euclidean distance, keep original scale without deliberate amplification
            M_cost = ot.dist(y_cf_np, y_target_np, metric='euclidean')
            
            # Non-one-hot marginal distributions: uniform distribution (total mass = 1, sum of each row = 1/N)
            a = ot.unif(N)  # [N], each element is 1/N, sum to 1
            b = ot.unif(M)  # [M], each element is 1/M, sum to 1
            
            # Calculate optimal transport matrix (N×M)
            nu = ot.sinkhorn(a, b, M_cost, reg=self.reg, numItermax=100000)
            
            # Convert back to torch tensor
            nu = torch.from_numpy(nu).to(y_cf.device)
            
            # # Print row sums (should be close to 1/N) and maximum weight of each row
            # print(f"Transport matrix row sums (should be close to 1/{N}): {nu.sum(dim=1)}")
            # print(f"Maximum weight of each row: {nu.max(dim=1).values}")

            for i in range(N):
                # Find the index of y_target corresponding to the maximum weight in each row
                max_weight_idx = torch.argmax(nu[i])
                max_weight_idx = torch.clamp(max_weight_idx, 0, M-1)  # Index safety
                
                selected_y_target = y_target[max_weight_idx]
                diff = torch.abs(y_cf[i] - selected_y_target)
                
                # print(f"y_cf[{i}] = {y_cf[i]:.4f}, matched target = {selected_y_target:.4f}, difference = {diff:.4f}")
                
                if diff > self.tau:
                    R_mask[i] = True

        G_mask = ~R_mask
        return R_mask, G_mask