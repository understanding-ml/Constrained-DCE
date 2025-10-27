import importlib
import torch

class ConstraintManager:
    """
    自动管理所有约束（torch版，返回torch.Tensor）
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
            特征列名顺序，用于在 torch.Tensor 里找到对应索引
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
                        feature_names=feature_names    # 🚩 必须显式传进去
                    )
                )

            elif ctype == "fsd":
                # FSDConstraintTorch 的 forward 接口不同，需要额外传 X_fact, X_cf, mu_list
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

    # def penalty(self, X: torch.Tensor, X_fact: torch.Tensor = None,
    #             X_cf: torch.Tensor = None, mu_list: list = None) -> torch.Tensor:
    #     """
    #     计算所有约束的惩罚
    #     - mean/std/lsc: 只用 X
    #     - fsd: 需要 (X_fact, X_cf, mu_list)
    #     返回: torch.Tensor (scalar, 可反向传播)
    #     """
    #     total_penalty = X.new_tensor(0.0)

    #     for c in self.constraints:
    #         if c.__class__.__name__ == "FSDConstraintTorch":
    #             if X_fact is None or X_cf is None or mu_list is None:
    #                 raise ValueError("FSDConstraintTorch requires X_fact, X_cf and mu_list")
    #             total_penalty = total_penalty + c(X_fact, X_cf, mu_list)
    #         else:
    #             total_penalty = total_penalty + c(X)

    #     return total_penalty.squeeze()
            

    # def penalty(self, X_fact=None, X_cf=None, mu_list=None):
    #     total_penalty = torch.tensor(0.0, device=X_fact.device if X_fact is not None else X_cf.device)
        
    #     for constraint in self.constraints:
    #         # 有些约束只用 X_cf，比如 Mean / STD
    #         if hasattr(constraint, "forward"):
    #             if "X_fact" in constraint.forward.__code__.co_varnames:
    #                 penalty_val = constraint.forward(X_fact, X_cf, mu_list)
    #             else:
    #                 penalty_val = constraint.forward(X_cf)  # 老接口兼容
    #         else:
    #             penalty_val = torch.tensor(0.0, device=total_penalty.device)
            
    #         total_penalty = total_penalty + penalty_val
        
    #     return total_penalty.squeeze()
            
#————————————-0928为了适配FSD SSD的比例抽样————————
    def penalty(self, X_fact=None, X_cf=None, mu_list=None):
        """
        统一计算所有约束的 penalty
        - mean/std/lsc: 只依赖 X_cf
        - fsd/ssd: 需要 X_fact, X_cf, mu_list
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



    # def debug(self, X: torch.Tensor, X_fact: torch.Tensor = None,
    #           X_cf: torch.Tensor = None, mu_list: list = None):
    #     """
    #     打印每个约束的 penalty + 梯度贡献，以及最终总和
    #     """
    #     total_penalty = 0.0
    #     total_grad = torch.zeros_like(X)

    #     for c in self.constraints:
    #         # clone 数据，避免梯度叠加
    #         X_clone = X.clone().detach().requires_grad_(True)

    #         if c.__class__.__name__ in ["FSDConstraintTorch", "SSDConstraintTorch"]:
    #             if X_fact is None or X_cf is None or mu_list is None:
    #                 raise ValueError(f"{c.__class__.__name__} requires X_fact, X_cf and mu_list")
    #             p = c(X_fact, X_cf, mu_list)
    #         else:
    #             p = c(X_clone)


    #         p.backward()

    #         grad = X_clone.grad

    #         print(f"[{c.__class__.__name__}] penalty={p.item():.4f}")
    #         print(f"[{c.__class__.__name__}] grad wrt X:\n{grad}\n")

    #         total_penalty += p.item()
    #         total_grad += grad

    #     print("====== TOTAL ======")
    #     print(f"Total penalty={total_penalty:.4f}")
    #     print(f"Total grad wrt X:\n{total_grad}")
