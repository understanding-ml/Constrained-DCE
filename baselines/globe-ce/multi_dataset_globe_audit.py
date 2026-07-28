import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from explainers.constraints.FSDConstraint import FSDConstraintTorch
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from explainers.globe_ce import GLOBE_CE
from models.mlp import BlackBoxModel


SEEDS = [40, 41, 42, 43, 44]
TRAIN_SEED = 42
SAMPLE_SIZE = 100
N_PROJECTIONS = 10
TRIM_DELTA = 0.1
GLOBE_DIRECTION_SAMPLES = 5000


DATASET_CONFIGS = {
    "Cardio": {
        "dce_file": "cardio/cardio_DCE.xlsx",
        "metric_sheet": "DCE_Metrics_Summary",
        "target": "cardio",
        "desired_label": 0,
        "beta": (0.1, 0.9),
        "mean": ("weight", 75.0, "lower"),
        "std": ("ap_lo", 38.0, "upper"),
        "lsc": None,
        "fsd": ("weight", -1),
    },
    "HELOC": {
        "dce_file": "HELOC/HELOC_DCE.xlsx",
        "metric_sheet": "DCE_Metrics_Summary",
        "target": "RiskPerformance",
        "desired_label": 0,
        "beta": (0.1, 0.9),
        "mean": ("AverageMInFile", 80.0, "upper"),
        "std": ("ExternalRiskEstimate", 10.0, "upper"),
        "lsc": ("NumSatisfactoryTrades", "NumTotalTrades", 0.8464),
        "fsd": ("NetFractionInstallBurden", -1),
    },
    "Market": {
        "dce_file": "market/market_DCE.xlsx",
        "metric_sheet": "DCE_Metrics_Summary",
        "target": "Response",
        "desired_label": 1,
        "beta": (0.5, 0.5),
        "mean": ("Recency", 54.0, "lower"),
        "std": ("Income", 19000.0, "upper"),
        "lsc": ("MntFruits", "MntSweetProducts", 0.5785),
        "fsd": ("NumDealsPurchases", 1),
    },
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_cardio(root):
    frame = pd.read_csv(root / "data/cardio/cardio.csv", delimiter=";")
    X = frame.iloc[:, 1:].drop(columns=["cardio"]).copy()
    y = frame["cardio"].copy()
    return X, y


def _load_heloc(root):
    frame = pd.read_csv(root / "data/HELOC/heloc_dataset_v1 (1).csv")
    frame = frame[frame["MSinceOldestTradeOpen"] != -9].copy()
    y = frame["RiskPerformance"].replace({"Good": 0, "Bad": 1})
    X = frame.drop(columns=["RiskPerformance"]).copy()
    return X, y


def _load_market(root):
    frame = pd.read_excel(root / "data/Marketing Campaign/marketing_campaign.xlsx")
    frame = frame.drop(columns=["Z_CostContact", "Z_Revenue"])
    education = {"PhD": 3, "Master": 2, "2n Cycle": 2, "Graduation": 1, "Basic": 0}
    marital = {"Married": 1, "Together": 1, "Single": 0, "Divorced": 0,
               "Alone": 0, "YOLO": 0, "Absurd": 0}
    encoder = LabelEncoder()
    for column in frame.columns:
        if column != "Response" and frame[column].dtype == "object":
            if column == "Education":
                frame[column] = frame[column].map(education)
            elif column == "Marital_Status":
                frame[column] = frame[column].map(marital)
            else:
                frame[column] = encoder.fit_transform(frame[column].astype(str))
    for column in frame.columns:
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(frame[column].median()).astype(int)
    X = frame.iloc[:, 1:].drop(columns=["Response"]).copy()
    y = frame["Response"].copy()
    return X, y


def prepare_dataset(name, root):
    loaders = {"Cardio": _load_cardio, "HELOC": _load_heloc, "Market": _load_market}
    X, y = loaders[name](root)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=TRAIN_SEED
    )
    mean = X_train_raw.mean()
    std = X_train_raw.std().replace(0, 1)
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    if name == "Cardio":
        X_test = X_test[X_test["ap_hi"] > X_test["ap_lo"]]
    indices = X_test.sample(SAMPLE_SIZE, random_state=TRAIN_SEED).index
    factual_norm = X_test.loc[indices].copy()
    factual_raw = factual_norm * std + mean
    return X_train, y_train, X_test, y_test, factual_norm, factual_raw, mean, std


def train_model(X_train, y_train):
    set_seed(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlackBoxModel(input_dim=X_train.shape[1]).to(device)
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).view(-1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    return model.cpu()


class DesiredOutcomeModel:
    def __init__(self, model, desired_label):
        self.model = model
        self.desired_label = desired_label

    def predict(self, X):
        with torch.no_grad():
            scores = self.model(torch.as_tensor(X, dtype=torch.float32)).reshape(-1)
        labels = (scores >= 0.5).to(torch.int64)
        return (labels == self.desired_label).to(torch.int64).numpy()


class DatasetMetadata:
    def __init__(self, name, features):
        self.name = name.lower()
        self.features_tree = {feature: [] for feature in features}
        self.features = list(features) + ["Outcome"]
        self.categorical_features = {self.name: []}
        self.continuous_features = {self.name: list(features)}


def _distribution_metrics(factual_norm, cf_norm, scores, targets, seed):
    set_seed(seed)
    swd = SlicedWassersteinDivergence(factual_norm.shape[1], N_PROJECTIONS)
    wd = WassersteinDivergence()
    factual = torch.tensor(factual_norm.values, dtype=torch.float32)
    counterfactual = torch.tensor(cf_norm.values, dtype=torch.float32)
    score_tensor = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)
    target_tensor = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    x_swd = np.sqrt(swd.distance(counterfactual, factual, TRIM_DELTA)[0].item())
    # The three original multi-run scripts store Y-WD without the outer square root.
    y_wd = wd.distance(score_tensor, target_tensor, TRIM_DELTA)[0].item()
    return float(x_swd), float(y_wd)


def _constraint_metrics(config, factual_raw, cf_raw, train_mean, train_std):
    output = {}
    mean_feature, _, _ = config["mean"]
    std_feature, _, _ = config["std"]
    output["Mean value"] = float(cf_raw[mean_feature].mean())
    # Match the validation DCE/CDCE analysis notebooks, which use
    # pandas Series.std() (sample standard deviation, ddof=1).
    output["Std value"] = float(cf_raw[std_feature].std(ddof=1))
    if config["lsc"] is None:
        output["LSC MSE"] = np.nan
    else:
        lhs, rhs, coefficient = config["lsc"]
        lhs_norm = (cf_raw[lhs] - train_mean[lhs]) / train_std[lhs]
        rhs_norm = (cf_raw[rhs] - train_mean[rhs]) / train_std[rhs]
        output["LSC MSE"] = float(np.mean((lhs_norm - coefficient * rhs_norm) ** 2))
    fsd_feature, direction = config["fsd"]
    fsd = FSDConstraintTorch.check_FSD(
        factual_raw[fsd_feature].to_numpy(), cf_raw[fsd_feature].to_numpy(), direction=direction
    )
    output["FSD violations"] = int(len(fsd["violations"]))
    return output


def _domain_metrics(X_train, factual_raw, cf_raw):
    changed = (cf_raw - factual_raw).abs().max(axis=1) > 1e-8
    below = (cf_raw < X_train.min()).any(axis=1)
    above = (cf_raw > X_train.max()).any(axis=1)
    integer_columns = [c for c in cf_raw if np.allclose(X_train[c], np.round(X_train[c]))]
    noninteger = (np.abs(cf_raw[integer_columns] - np.round(cf_raw[integer_columns])) > 1e-8).any(axis=1)
    invalid = changed & (below | above | noninteger)
    return {
        "Changed count": int(changed.sum()),
        "Invalid changed count": int(invalid.sum()),
        "Valid changed rate": float(1.0 - invalid.sum() / max(changed.sum(), 1)),
        "Out-of-range count": int((changed & (below | above)).sum()),
        "Non-integer count": int((changed & noninteger).sum()),
    }


def run_globe_dataset(name, root, seeds=SEEDS):
    config = DATASET_CONFIGS[name]
    X_train, y_train, _, _, factual_norm, factual_raw, mean, std = prepare_dataset(name, root)
    model = train_model(X_train, y_train)
    wrapped_model = DesiredOutcomeModel(model, config["desired_label"])
    metadata = DatasetMetadata(name, factual_norm.columns)
    rows = []
    counterfactuals = {}
    for seed in seeds:
        set_seed(seed)
        beta_a, beta_b = config["beta"]
        # Recreate the fixed target distribution used by the original dataset script.
        set_seed(TRAIN_SEED)
        targets = torch.distributions.beta.Beta(beta_a, beta_b).sample((SAMPLE_SIZE,)).numpy()
        set_seed(seed)
        globe = GLOBE_CE(wrapped_model, metadata, factual_norm, normalise=None, p=1)
        start = time.perf_counter()
        globe.sample(n_sample=GLOBE_DIRECTION_SAMPLES, magnitude=1.0, sparsity_power=1.0,
                     disable_tqdm=True, seed=seed, scheme="random")
        _, scaled_costs, scalars = globe.scale(
            globe.best_delta, scalars="auto", n_scalars=1000, vector=True, disable_tqdm=True
        )
        _, success_indices = globe.min_scalar_costs(
            scaled_costs, return_idxs=True, remove_nan=False
        )
        affected_cf = globe.x_aff.copy()
        successful = ~np.isnan(success_indices)
        for row_idx in np.where(successful)[0]:
            affected_cf[row_idx] = (
                globe.x_aff[row_idx] + scalars[int(success_indices[row_idx])] * globe.best_delta
            )
        affected_mask = wrapped_model.predict(factual_norm.values) == 0
        cf_norm = factual_norm.copy()
        cf_norm.loc[affected_mask, :] = affected_cf
        cf_raw = cf_norm * std + mean
        with torch.no_grad():
            scores = model(torch.tensor(cf_norm.values, dtype=torch.float32)).reshape(-1).numpy()
        labels = (scores >= 0.5).astype(int)
        favorable = labels == config["desired_label"]
        x_swd, y_wd = _distribution_metrics(factual_norm, cf_norm, scores, targets, seed)
        row = {
            "Dataset": name,
            "Method": "GLOBE-CE",
            "seed": seed,
            "Affected count": int(len(globe.x_aff)),
            "Successful recourse": int(successful.sum()),
            "Coverage": float(successful.mean()) if len(successful) else 1.0,
            "Favorable rate": float(favorable.mean()),
            "SWD": x_swd,
            "WD": y_wd,
            "Runtime (s)": float(time.perf_counter() - start),
        }
        row.update(_constraint_metrics(config, factual_raw, cf_raw, mean, std))
        row.update(_domain_metrics(X_train * std + mean, factual_raw, cf_raw))
        rows.append(row)
        counterfactuals[seed] = cf_raw.assign(model_score=scores, favorable=favorable)
    return pd.DataFrame(rows), counterfactuals


def load_existing_alignment(root):
    rows = []
    constrained_files = {
        "Cardio": {"Mean": "cardio/cardio_MeanConstraint.xlsx", "Std": "cardio/cardio_StdConstraint.xlsx",
                   "LSC": "cardio/cardio_LSCConstraint.xlsx", "FSD": "cardio/cardio_FSDConstraint.xlsx"},
        "HELOC": {"Mean": "HELOC/HELOC_MeanConstraint.xlsx", "Std": "HELOC/HELOC_StdConstraint.xlsx",
                  "LSC": "HELOC/HELOC_LSCConstraint.xlsx", "FSD": "HELOC/HELOC_FSDConstraint_NetFractionInstallBurden_0.005.xlsx"},
        "Market": {"Mean": "market/market_MeanConstraint.xlsx", "Std": "market/market_StdConstraint.xlsx",
                  "LSC": "market/market_LSCConstraint.xlsx", "FSD": "market/market_FSDConstraint_NumDealsPurchases_0.004.xlsx"},
    }
    for name, config in DATASET_CONFIGS.items():
        dce = pd.read_excel(root / config["dce_file"], sheet_name=config["metric_sheet"])
        for _, item in dce.iterrows():
            rows.append({"Dataset": name, "Method": "DCE", "seed": int(item["seed"]),
                         "SWD": float(item["X_SWD"]), "WD": float(item["Y_WD"])})
        for constraint, filename in constrained_files[name].items():
            path = root / filename
            if not path.exists():
                continue
            sheets = pd.ExcelFile(path).sheet_names
            metric_sheets = [s for s in sheets if "Metrics_Summary" in s]
            if not metric_sheets:
                continue
            metrics = pd.read_excel(path, sheet_name=metric_sheets[0])
            swd_col = next(c for c in metrics if "X_SWD" in c)
            wd_col = next(c for c in metrics if "Y_WD" in c)
            for _, item in metrics.iterrows():
                rows.append({"Dataset": name, "Method": f"CDCE-{constraint}",
                             "seed": int(item["seed"]), "SWD": float(item[swd_col]),
                             "WD": float(item[wd_col])})
    return pd.DataFrame(rows)


def summarize_runs(frame):
    numeric = [c for c in frame.select_dtypes(include=np.number).columns if c != "seed"]
    summary = frame.groupby(["Dataset", "Method"])[numeric].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def run_all(root=".", output_dir="multi_dataset_globe_results"):
    root = Path(root).resolve()
    output = root / output_dir
    output.mkdir(exist_ok=True)
    all_globe = []
    all_counterfactuals = {}
    for name in DATASET_CONFIGS:
        runs, counterfactuals = run_globe_dataset(name, root)
        all_globe.append(runs)
        all_counterfactuals[name] = counterfactuals
    globe_runs = pd.concat(all_globe, ignore_index=True)
    existing = load_existing_alignment(root)
    alignment_runs = pd.concat([
        existing,
        globe_runs[["Dataset", "Method", "seed", "SWD", "WD"]],
    ], ignore_index=True)
    globe_summary = summarize_runs(globe_runs)
    alignment_summary = summarize_runs(alignment_runs)
    globe_runs.to_csv(output / "globe_ce_5runs.csv", index=False)
    globe_summary.to_csv(output / "globe_ce_5runs_summary.csv", index=False)
    alignment_runs.to_csv(output / "dce_cdce_globe_alignment_5runs.csv", index=False)
    alignment_summary.to_csv(output / "dce_cdce_globe_alignment_summary.csv", index=False)
    with pd.ExcelWriter(output / "multi_dataset_globe_ce_audit.xlsx") as writer:
        globe_runs.to_excel(writer, sheet_name="GLOBE_runs", index=False)
        globe_summary.to_excel(writer, sheet_name="GLOBE_summary", index=False)
        alignment_runs.to_excel(writer, sheet_name="Alignment_runs", index=False)
        alignment_summary.to_excel(writer, sheet_name="Alignment_summary", index=False)
        for name, seed_frames in all_counterfactuals.items():
            for seed, frame in seed_frames.items():
                frame.to_excel(writer, sheet_name=f"{name[:6]}_{seed}", index=False)
    return globe_runs, globe_summary, alignment_runs, alignment_summary
