import pandas as pd
import torch
from typing import Sequence, Tuple, Dict

def recover_original_df(
    best_X: torch.Tensor,
    explain_columns: Sequence[str],
    mean: pd.Series,
    std: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将 best_X（归一化/标准化空间的反事实样本）恢复到原始量纲。
    返回:
      df_normalized: 归一化空间下的 DataFrame
      df_original  : 原始量纲下的 DataFrame
    说明:
      若你的归一化方式为 x_norm = (x_raw - mean) / std，则逆变换为:
      x_raw = x_norm * std + mean
    """
    df_normalized = pd.DataFrame(
        best_X.detach().cpu().numpy(),
        columns=list(explain_columns)
    )
    df_original = df_normalized * std[explain_columns] + mean[explain_columns]
    return df_normalized, df_original

def column_stats(
    df_normalized: pd.DataFrame,
    df_original: pd.DataFrame,
    col: str
) -> Dict[str, float]:
    """
    打印并返回指定列在归一化/原始量纲下的均值与标准差（ddof=0）。
    """
    mean_norm = float(df_normalized[col].mean())
    std_norm  = float(df_normalized[col].std(ddof=0))
    mean_raw  = float(df_original[col].mean())
    std_raw   = float(df_original[col].std(ddof=0))

    print(f"[{col}] normalized: mean={mean_norm:.4f}, std={std_norm:.4f}")
    print(f"[{col}] original  : mean={mean_raw:.4f}, std={std_raw:.4f}")

    return {
        "mean_norm": mean_norm, "std_norm": std_norm,
        "mean_raw": mean_raw,   "std_raw": std_raw
    }
