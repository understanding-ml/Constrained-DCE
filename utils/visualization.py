import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde
import torch
from statsmodels.distributions.empirical_distribution import ECDF

def plot_quantile(factual, counterfactual, column_name):
    quantiles_factual = factual[column_name].quantile(np.linspace(0, 1, 100))
    quantiles_counterfactual = counterfactual[column_name].quantile(
        np.linspace(0, 1, 100)
    )

    # Plot quantiles
    plt.figure(figsize=(8, 6))
    plt.plot(quantiles_factual.values, np.linspace(0, 1, 100), label="Factual")
    plt.plot(
        quantiles_counterfactual.values, np.linspace(0, 1, 100), label="Counterfactual"
    )
    plt.xlabel("Feature Values")
    plt.ylabel("Quantiles")
    plt.title(f"{column_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_box_whisker(factual, counterfactual, column_name):
    # Combine the data into a single DataFrame for plotting
    data_to_plot = pd.DataFrame(
        {"Factual": factual[column_name], "Counterfactual": counterfactual[column_name]}
    )

    # Creating the boxplot
    plt.figure(figsize=(10, 6))
    data_to_plot.boxplot(column=["Factual", "Counterfactual"])
    plt.title("Box-and-Whisker Plot for Factual and Counterfactual")
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()


def category_box_plot(df, x, y, hue, title):
    plt.figure(figsize=(14, 12))

    plt.subplot(212)
    g2 = sns.boxplot(x=x, y=y, data=df.sort_values(by=x), palette="hls", hue=hue)
    g2.set_xlabel(x, fontsize=12)
    g2.set_ylabel(y, fontsize=12)
    g2.set_title(title, fontsize=15)

    plt.subplots_adjust(hspace=0.6, top=0.8)

    plt.show()


def hist_plot(df, x, hue, title):
    plt.figure(figsize=(14, 12))

    plt.subplot(221)
    g = sns.countplot(x=x, data=df.sort_values(by=x), palette="hls", hue=hue)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_xlabel(x, fontsize=12)
    g.set_ylabel("Count", fontsize=12)
    g.set_title(title, fontsize=20)

#-----0929新增：对比指定维度在 事实分布、不带约束的反事实、带约束的反事实 下的 PDF，并在图例中标注对应的均值和方差。#

# def plot_pdf_column(datasets, column_name, labels=None, mean=None, std=None, explain_columns=None):
#     """
#     绘制多个数据集在指定列上的 PDF 曲线，并标注均值和标准差。
    
#     参数:
#         datasets: list[pd.DataFrame/np.ndarray/torch.Tensor]  
#             1–N 个数据集，可以是 DataFrame / ndarray / Tensor
#         column_name: str  
#             需要绘制的列名
#         labels: list[str] 或 None  
#             图例名称，可选。如果不传则自动命名为 Dataset 1, 2, ...
#         mean: float 或 None  
#             归一化时的均值，用于逆归一化
#         std: float 或 None  
#             归一化时的标准差，用于逆归一化
#         explain_columns: list[str] 或 None  
#             当输入为 ndarray / Tensor 时，需要提供列名列表，用来找到 column_name 的索引

#     示例:
#         plot_pdf_column(
#             [df_explain, X_s, X_s_ssd],
#             "Credit amount",
#             labels=["Factual", "No constraint", "SSD constraint"],
#             mean=3360.6, std=2000.5,
#             explain_columns=explain_columns
#         )
#     """
#     if labels is None:
#         labels = [f"Dataset {i+1}" for i in range(len(datasets))]
#     elif len(labels) != len(datasets):
#         raise ValueError("labels 数量必须与 datasets 数量一致，或不传 labels")

#     values_list = []
#     for data in datasets:
#         if isinstance(data, pd.DataFrame):
#             values = data[column_name].values
#         elif isinstance(data, np.ndarray):
#             if explain_columns is None:
#                 raise ValueError("ndarray 输入必须提供 explain_columns")
#             idx = explain_columns.index(column_name)
#             values = data[:, idx]
#         elif isinstance(data, torch.Tensor):
#             if explain_columns is None:
#                 raise ValueError("Tensor 输入必须提供 explain_columns")
#             idx = explain_columns.index(column_name)
#             values = data[:, idx].cpu().numpy()
#         else:
#             raise TypeError("datasets 必须是 DataFrame / ndarray / Tensor 之一")
#         values_list.append(values)

#     # 全局范围
#     all_values = np.concatenate(values_list)
#     if mean is not None and std is not None:
#         all_values = all_values * std + mean
#         values_list = [v * std + mean for v in values_list]

#     x_min, x_max = all_values.min(), all_values.max()
#     x_vals = np.linspace(x_min, x_max, 300)

#     plt.figure(figsize=(6, 4))
#     text_info = []
#     for values, label in zip(values_list, labels):
#         pdf = gaussian_kde(values)(x_vals)
#         m = np.mean(values)
#         s = np.std(values)

#         plt.plot(x_vals, pdf, label=label)
#         plt.axvline(m, color=plt.gca().lines[-1].get_color(),
#                     linestyle="--", alpha=0.7)

#         text_info.append(f"{label}: mean={m:.3f}, std={s:.3f}")

#     plt.text(0.98, 0.95, "\n".join(text_info),
#              transform=plt.gca().transAxes,
#              fontsize=10, va="top", ha="right",
#              bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"))

#     plt.xlabel(column_name)
#     plt.ylabel("PDF")
#     plt.title(f"PDF comparison on {column_name}")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.7)
#     plt.tight_layout()
#     plt.show()
    

#--------------1004,支持PDF CDF CDF积分-------------------#


# def plot_pdf_column(datasets, column_name, labels=None, mean=None, std=None, explain_columns=None, smooth=False):
#     """
#     绘制多个数据集在指定列上的 PDF 曲线，并标注均值和标准差。
    
#     参数:
#         datasets: list[pd.DataFrame/np.ndarray/torch.Tensor]  
#             1–N 个数据集，可以是 DataFrame / ndarray / Tensor
#         column_name: str  
#             需要绘制的列名
#         labels: list[str] 或 None  
#             图例名称，可选。如果不传则自动命名为 Dataset 1, 2, ...
#         mean: float 或 None  
#             归一化时的均值，用于逆归一化
#         std: float 或 None  
#             归一化时的标准差，用于逆归一化
#         explain_columns: list[str] 或 None  
#             当输入为 ndarray / Tensor 时，需要提供列名列表，用来找到 column_name 的索引
#         smooth: bool  
#             是否对 PDF 进行平滑，默认值为 False（不平滑）。如果为 True，则对 PDF 进行平滑。
    
#     示例:
#         plot_pdf_column(
#             [df_explain, X_s, X_s_ssd],
#             "Credit amount",
#             labels=["Factual", "No constraint", "SSD constraint"],
#             mean=3360.6, std=2000.5,
#             explain_columns=explain_columns,
#             smooth=True
#         )
#     """
#     if labels is None:
#         labels = [f"Dataset {i+1}" for i in range(len(datasets))]
#     elif len(labels) != len(datasets):
#         raise ValueError("labels 数量必须与 datasets 数量一致，或不传 labels")

#     values_list = []
#     for data in datasets:
#         if isinstance(data, pd.DataFrame):
#             values = data[column_name].values
#         elif isinstance(data, np.ndarray):
#             if explain_columns is None:
#                 raise ValueError("ndarray 输入必须提供 explain_columns")
#             idx = explain_columns.index(column_name)
#             values = data[:, idx]
#         elif isinstance(data, torch.Tensor):
#             if explain_columns is None:
#                 raise ValueError("Tensor 输入必须提供 explain_columns")
#             idx = explain_columns.index(column_name)
#             values = data[:, idx].cpu().numpy()
#         else:
#             raise TypeError("datasets 必须是 DataFrame / ndarray / Tensor 之一")
#         values_list.append(values)

#     # 全局范围
#     all_values = np.concatenate(values_list)
#     if mean is not None and std is not None:
#         all_values = all_values * std + mean
#         values_list = [v * std + mean for v in values_list]

#     x_min, x_max = all_values.min(), all_values.max()
#     x_vals = np.linspace(x_min, x_max, 300)

#     plt.figure(figsize=(6, 4))
#     text_info = []
#     for values, label in zip(values_list, labels):
#         if smooth:
#             # 使用 KDE 进行平滑处理
#             pdf = gaussian_kde(values)(x_vals)
#         else:
#             # 直接计算 PDF
#             hist, bin_edges = np.histogram(values, bins=30, density=True)
#             pdf = np.interp(x_vals, bin_edges[:-1], hist)

#         # 绘制平滑后的 PDF
#         m = np.mean(values)
#         s = np.std(values)

#         plt.plot(x_vals, pdf, label=label)
#         plt.axvline(m, color=plt.gca().lines[-1].get_color(),
#                     linestyle="--", alpha=0.7)

#         text_info.append(f"{label}: mean={m:.3f}, std={s:.3f}")

#     plt.text(0.98, 0.95, "\n".join(text_info),
#              transform=plt.gca().transAxes,
#              fontsize=10, va="top", ha="right",
#              bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"))

#     plt.xlabel(column_name)
#     plt.ylabel("PDF")
#     plt.title(f"PDF comparison on {column_name}")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.7)
#     plt.tight_layout()
#     plt.show()



def plot_pdf_column(datasets, column_name, labels=None, mean=None, std=None, 
                   explain_columns=None, use_smoothing=False):
    """
    绘制多个数据集在指定列上的 PDF 曲线，并标注均值和标准差。
    
    参数:
        datasets: list[pd.DataFrame/np.ndarray/torch.Tensor]  
            1–N 个数据集，可以是 DataFrame / ndarray / Tensor
        column_name: str  
            需要绘制的列名
        labels: list[str] 或 None  
            图例名称，可选。如果不传则自动命名为 Dataset 1, 2, ...
        mean: float 或 None  
            归一化时的均值，用于逆归一化
        std: float 或 None  
            归一化时的标准差，用于逆归一化
        explain_columns: list[str] 或 None  
            当输入为 ndarray / Tensor 时，需要提供列名列表，用来找到 column_name 的索引
        use_smoothing: bool  
            是否使用平滑（KDE），False则使用纯直方图，默认False
    """
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]
    elif len(labels) != len(datasets):
        raise ValueError("labels 数量必须与 datasets 数量一致，或不传 labels")

    values_list = []
    for data in datasets:
        if isinstance(data, pd.DataFrame):
            values = data[column_name].values
        elif isinstance(data, np.ndarray):
            if explain_columns is None:
                raise ValueError("ndarray 输入必须提供 explain_columns")
            idx = explain_columns.index(column_name)
            values = data[:, idx]
        elif isinstance(data, torch.Tensor):
            if explain_columns is None:
                raise ValueError("Tensor 输入必须提供 explain_columns")
            idx = explain_columns.index(column_name)
            values = data[:, idx].cpu().numpy()
        else:
            raise TypeError("datasets 必须是 DataFrame / ndarray / Tensor 之一")
        values_list.append(values)

    # 处理逆归一化
    if mean is not None and std is not None:
        values_list = [v * std + mean for v in values_list]
        all_values = np.concatenate(values_list)
    else:
        all_values = np.concatenate(values_list)

    # 确定数据范围并适当扩展
    x_min, x_max = all_values.min(), all_values.max()
    x_range = x_max - x_min
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    x_vals = np.linspace(x_min, x_max, 300)

    plt.figure(figsize=(6, 4))
    text_info = []
    
    # 为所有数据集使用相同的分箱（仅用于非平滑模式）
    bins = np.linspace(x_min, x_max, 30)
    bin_width = bins[1] - bins[0]

    for values, label in zip(values_list, labels):
        if use_smoothing:
            # 平滑模式：使用KDE
            pdf = gaussian_kde(values)(x_vals)
        else:
            # 非平滑模式：使用直方图
            counts, _ = np.histogram(values, bins=bins)
            pdf = counts / (len(values) * bin_width)
            # 基于分箱中点插值绘制连续曲线
            bin_centers = (bins[:-1] + bins[1:]) / 2
            pdf = np.interp(x_vals, bin_centers, pdf)
        
        # 计算均值和标准差
        m = np.mean(values)
        s = np.std(values)

        # 绘制PDF曲线
        plt.plot(x_vals, pdf, label=label)
        # 绘制均值线
        plt.axvline(m, color=plt.gca().lines[-1].get_color(),
                    linestyle="--", alpha=0.7)

        text_info.append(f"{label}: mean={m:.3f}, std={s:.3f}")

    # 添加统计信息文本框
    plt.text(0.98, 0.95, "\n".join(text_info),
             transform=plt.gca().transAxes,
             fontsize=10, va="top", ha="right",
             bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"))

    plt.xlabel(column_name)
    plt.ylabel("PDF")
    plt.title(f"PDF comparison on {column_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    






def plot_cdf_column(datasets, column_name, labels=None, mean=None, std=None, explain_columns=None, smooth=False):
    """
    绘制多个数据集在指定列上的 CDF 曲线，并标注均值和标准差。
    
    参数:
        datasets: list[pd.DataFrame/np.ndarray/torch.Tensor]  
            1–N 个数据集，可以是 DataFrame / ndarray / Tensor
        column_name: str  
            需要绘制的列名
        labels: list[str] 或 None  
            图例名称，可选。如果不传则自动命名为 Dataset 1, 2, ...
        mean: float 或 None  
            归一化时的均值，用于逆归一化
        std: float 或 None  
            归一化时的标准差，用于逆归一化
        explain_columns: list[str] 或 None  
            当输入为 ndarray / Tensor 时，需要提供列名列表，用来找到 column_name 的索引
        smooth: bool  
            是否对 CDF 进行平滑，默认值为 False（不平滑）。如果为 True，则对 CDF 进行平滑。
    
    示例:
        plot_cdf_column(
            [df_explain, X_s, X_s_ssd],
            "Credit amount",
            labels=["Factual", "No constraint", "SSD constraint"],
            mean=3360.6, std=2000.5,
            explain_columns=explain_columns,
            smooth=True
        )
    """
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]
    elif len(labels) != len(datasets):
        raise ValueError("labels 数量必须与 datasets 数量一致，或不传 labels")

    values_list = []
    for data in datasets:
        if isinstance(data, pd.DataFrame):
            values = data[column_name].values
        elif isinstance(data, np.ndarray):
            if explain_columns is None:
                raise ValueError("ndarray 输入必须提供 explain_columns")
            idx = explain_columns.index(column_name)
            values = data[:, idx]
        elif isinstance(data, torch.Tensor):
            if explain_columns is None:
                raise ValueError("Tensor 输入必须提供 explain_columns")
            idx = explain_columns.index(column_name)
            values = data[:, idx].cpu().numpy()
        else:
            raise TypeError("datasets 必须是 DataFrame / ndarray / Tensor 之一")
        values_list.append(values)

    # 全局范围
    all_values = np.concatenate(values_list)
    if mean is not None and std is not None:
        all_values = all_values * std + mean
        values_list = [v * std + mean for v in values_list]

    x_min, x_max = all_values.min(), all_values.max()
    x_vals = np.linspace(x_min, x_max, 300)

    plt.figure(figsize=(6, 4))
    text_info = []
    for values, label in zip(values_list, labels):
        values_sorted = np.sort(values)
        cdf = np.linspace(0, 1, len(values_sorted))

        if smooth:
            # 使用 KDE 对 CDF 进行平滑
            cdf_smooth = gaussian_kde(values)(x_vals)
            plt.plot(x_vals, cdf_smooth, label=label)
        else:
            plt.plot(values_sorted, cdf, label=label)

        m = np.mean(values)
        s = np.std(values)

        plt.axvline(m, color=plt.gca().lines[-1].get_color(),
                    linestyle="--", alpha=0.7)

        text_info.append(f"{label}: mean={m:.3f}, std={s:.3f}")

    plt.text(0.98, 0.95, "\n".join(text_info),
             transform=plt.gca().transAxes,
             fontsize=10, va="top", ha="right",
             bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"))

    plt.xlabel(column_name)
    plt.ylabel("CDF")
    plt.title(f"CDF comparison on {column_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

def plot_integral_cdf_column(datasets, column_name, labels=None, mean=None, std=None, explain_columns=None):
    """
    绘制多个数据集在指定列上的 CDF 积分曲线（不使用平滑）。
    
    参数:
        datasets: list[pd.DataFrame/np.ndarray/torch.Tensor]  
            1–N 个数据集，可以是 DataFrame / ndarray / Tensor
        column_name: str  
            需要绘制的列名
        labels: list[str] 或 None  
            图例名称，可选。如果不传则自动命名为 Dataset 1, 2, ...
        mean: float 或 None  
            归一化时的均值，用于逆归一化
        std: float 或 None  
            归一化时的标准差，用于逆归一化
        explain_columns: list[str] 或 None  
            当输入为 ndarray / Tensor 时，需要提供列名列表，用来找到 column_name 的索引
    """
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]
    elif len(labels) != len(datasets):
        raise ValueError("labels 数量必须与 datasets 数量一致，或不传 labels")

    # 为每个数据集绘制图形
    plt.figure(figsize=(6, 4))

    for idx, data in enumerate(datasets):
        # 获取当前数据列
        if isinstance(data, pd.DataFrame):
            values = data[column_name].values
        elif isinstance(data, np.ndarray):
            if explain_columns is None:
                raise ValueError("ndarray 输入必须提供 explain_columns")
            idx_column = explain_columns.index(column_name)
            values = data[:, idx_column]
        elif isinstance(data, torch.Tensor):
            if explain_columns is None:
                raise ValueError("Tensor 输入必须提供 explain_columns")
            idx_column = explain_columns.index(column_name)
            values = data[:, idx_column].cpu().numpy()
        else:
            raise TypeError("datasets 必须是 DataFrame / ndarray / Tensor 之一")

        # 处理逆归一化
        if mean is not None and std is not None:
            values = values * std + mean

        # 对 values 进行排序
        values_sorted = np.sort(values)

        # 计算 CDF：分配每个数据点的概率为 1/N
        cdf = np.arange(1, len(values_sorted) + 1) / len(values_sorted)

        # 计算每两个相邻点之间的间距 dx
        dx = np.diff(values_sorted, prepend=values_sorted[0])

        # 手动计算 CDF 的积分：每个点的积分是它之前所有点的累计概率乘以 dx
        integral_cdf = np.zeros_like(cdf)
        for i in range(1, len(values_sorted)):
            # 对于每个点，计算它之前所有点的累计概率，并乘以 dx
            integral_cdf[i] = integral_cdf[i-1] + cdf[i] * dx[i]

        # 绘制 CDF 积分曲线
        plt.plot(values_sorted, integral_cdf, label=f"{labels[idx]}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

    plt.xlabel(column_name)
    plt.ylabel("Integral CDF")
    plt.title(f"Integral CDF comparison on {column_name}")

    # 显示图例
    plt.legend(loc="best")  # 使用 "best" 位置来避免遮挡内容

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()  # 自动调整布局，防止图例遮挡
    plt.show()
