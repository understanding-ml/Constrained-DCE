import importlib
import torch

class ConstraintManager:
    """
    Automatically manages all constraints (PyTorch version, returns torch.Tensor)
    """
    def __init__(self, configs: list, feature_names: list):
        """
        configs: list of dict
            [
                {"type": "mean", "bounds": {"Duration": (0,15)}, "lambda": 1.0},
                {"type": "std",  "bounds": {"Duration": 5.0},   "lambda": 2.0},
                {"type": "lsc",  "relation": {"Duration": (["Credit","Income"], [0.1,0.05], 5.0)}, 
                                "lambda": 1.0, "mode": "strict"},
                {"type": "fsd",  "dir_map": {"Duration": +1}, "lambda": 1.0}
            ]
        feature_names: list
            Order of feature column names, used to find corresponding indices in torch.Tensor
        """
        self.constraints = []
        for cfg in configs:
            ctype = cfg["type"].lower()
            lam = cfg.get("lambda", 1.0)

            if ctype == "mean":
                module = importlib.import_module("explainers.constraints.MeanConstraint")
                cls = getattr(module, "MeanConstraintTorch")
                self.constraints.append(
                    cls(bounds=cfg["bounds"], lam=lam, feature_names=feature_names)
                )

            elif ctype == "std":
                module = importlib.import_module("explainers.constraints.StdConstraint")
                cls = getattr(module, "StdConstraintTorch")
                self.constraints.append(
                    cls(bounds=cfg["bounds"], lam=lam, feature_names=feature_names)
                )

            elif ctype == "lsc":
                module = importlib.import_module("explainers.constraints.LSCConstraint")
                cls = getattr(module, "LSCConstraintTorch")
                self.constraints.append(
                    cls(
                        relation=cfg["relation"],
                        lam=lam,
                        mode=cfg.get("mode", "strict"),
                        tolerance=cfg.get("tolerance", None),
                        feature_names=feature_names   
                    )
                )

            elif ctype == "fsd":
                # FSDConstraintTorch has a different forward interface, requiring additional X_fact, X_cf, mu_list
                module = importlib.import_module("explainers.constraints.FSDConstraint")
                cls = getattr(module, "FSDConstraintTorch")
                self.constraints.append(
                    cls(lam=lam, feature_names=feature_names, dir_map=cfg["dir_map"])
                )
                
            elif ctype == "ssd":
                module = importlib.import_module("explainers.constraints.SSDConstraint")
                cls = getattr(module, "SSDConstraintTorch")
                self.constraints.append(
                    cls(lam=lam, feature_names=feature_names, dir_map=cfg["dir_map"],
                        M=cfg.get("M", None), sample_mode=cfg.get("sample_mode", "quantile"))
                )

            else:
                raise ValueError(f"Unknown constraint type: {ctype}")

            
#———————————— Adapt to proportional sampling for FSD and SSD ————————————
    def penalty(self, X_fact=None, X_cf=None, mu_list=None):
        """
        Unified calculation of penalties for all constraints
        - mean/std/lsc: only depend on X_cf
        - fsd/ssd: require X_fact, X_cf, mu_list
        """
        device = X_cf.device if X_cf is not None else X_fact.device
        total_penalty = torch.tensor(0.0, device=device)

        for constraint in self.constraints:
            cname = constraint.__class__.__name__

            if cname in ["FSDConstraintTorch", "SSDConstraintTorch"]:
                if X_fact is None or X_cf is None or mu_list is None:
                    raise ValueError(f"{cname} requires X_fact, X_cf, mu_list")
                penalty_val = constraint(X_fact, X_cf, mu_list)
            else:
                penalty_val = constraint(X_cf)

            total_penalty = total_penalty + penalty_val

        return total_penalty.squeeze()


