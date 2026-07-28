import importlib.util
import os
import random
import time
from pathlib import Path

import dice_ml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from explainers.constraints.FSDConstraint import FSDConstraintTorch
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from models.mlp import BlackBoxModel


SEEDS = [40, 41, 42, 43, 44]
TRAIN_SEED = 42
SAMPLE_SIZE = 100
N_PROJECTIONS = 10
TRIM_DELTA = 0.1
RANDOM_SAMPLE_SIZE = 2000


def _load_shared_module(root):
    module_path = root / "baselines" / "globe-ce" / "multi_dataset_globe_audit.py"
    spec = importlib.util.spec_from_file_location("shared_globe_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProbabilityAdapter:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        array = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            score = self.model(torch.tensor(array, dtype=torch.float32)).reshape(-1).numpy()
        return np.column_stack([1.0 - score, score])


def train_model(X_train, y_train):
    set_seed(TRAIN_SEED)
    model = BlackBoxModel(input_dim=X_train.shape[1])
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def prepare_german(root):
    frame = pd.read_csv(root / "data/german_credit/german_credit_data.csv")
    target = frame["Risk"].replace({"good": 0, "bad": 1})
    frame["Risk"] = target
    saving = {"little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    checking = {"little": 1, "moderate": 2, "rich": 3}
    housing = {"free": 0, "rent": 1, "own": 2}
    encoder = LabelEncoder()
    for column in frame.columns:
        if column != "Risk" and frame[column].dtype == "object":
            if column == "Saving accounts":
                frame[column] = frame[column].map(saving)
            elif column == "Checking account":
                frame[column] = frame[column].map(checking)
            elif column == "Housing":
                frame[column] = frame[column].map(housing)
            else:
                frame[column] = encoder.fit_transform(frame[column].astype(str))
    for column in frame.columns:
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(frame[column].median()).astype(int)
    features = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account",
                "Credit amount", "Duration", "Purpose"]
    X = frame[features].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=TRAIN_SEED
    )
    mean = X_train_raw.mean()
    std = X_train_raw.std().replace(0, 1)
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    indices = X_test.sample(SAMPLE_SIZE, random_state=TRAIN_SEED).index
    factual_norm = X_test.loc[indices].copy()
    factual_raw = X_test_raw.loc[indices].copy()
    return X_train, y_train, factual_norm, factual_raw, mean, std


def prepare_named_dataset(name, root):
    if name == "German Credit":
        return prepare_german(root)
    shared = _load_shared_module(root)
    X_train, y_train, _, _, factual_norm, factual_raw, mean, std = shared.prepare_dataset(name, root)
    return X_train, y_train, factual_norm, factual_raw, mean, std


def dataset_config(name, root):
    if name == "German Credit":
        return {
            "desired_label": 0,
            "beta": (0.1, 0.9),
            "mean": ("Duration", 21.0, "lower"),
            "std": ("Credit amount", 2000.0, "upper"),
            "lsc": ("Duration", "Credit amount", 0.6264),
            "fsd": ("Duration", -1),
            "sqrt_wd": True,
            # German Credit reports the population standard deviation.
            "std_ddof": 0,
        }
    shared = _load_shared_module(root)
    config = dict(shared.DATASET_CONFIGS[name])
    config["sqrt_wd"] = False
    # The three validation notebooks use pandas Series.std(), i.e. ddof=1.
    config["std_ddof"] = 1
    if name == "Cardio":
        config["lsc"] = ("cholesterol", "gluc", 0.4557)
    return config


def build_dice(X_train, y_train, model):
    outcome = "__outcome__"
    training = X_train.copy()
    training[outcome] = y_train.to_numpy()
    data = dice_ml.Data(
        dataframe=training,
        continuous_features=list(X_train.columns),
        outcome_name=outcome,
    )
    dice_model = dice_ml.Model(model=ProbabilityAdapter(model), backend="sklearn", model_type="classifier")
    return dice_ml.Dice(data, dice_model, method="random")


def generate_population(dice, model, factual_norm, desired_label, seed):
    with torch.no_grad():
        factual_scores = model(torch.tensor(factual_norm.values, dtype=torch.float32)).reshape(-1).numpy()
    factual_labels = (factual_scores >= 0.5).astype(int)
    affected_mask = factual_labels != desired_label
    counterfactual = factual_norm.copy()
    successful = np.zeros(len(factual_norm), dtype=bool)
    start = time.perf_counter()
    for position in np.where(affected_mask)[0]:
        query = factual_norm.iloc[[position]]
        try:
            result = dice.generate_counterfactuals(
                query,
                total_CFs=1,
                desired_class=desired_label,
                features_to_vary="all",
                posthoc_sparsity_param=0.0,
                sample_size=RANDOM_SAMPLE_SIZE,
                random_seed=seed * 1000 + int(position),
                verbose=False,
            )
            final = result.cf_examples_list[0].final_cfs_df
            if final is not None and len(final):
                counterfactual.iloc[position] = final.iloc[0][factual_norm.columns].astype(float)
                successful[position] = True
        except UserConfigValidationException:
            pass
    runtime = time.perf_counter() - start
    with torch.no_grad():
        scores = model(torch.tensor(counterfactual.values, dtype=torch.float32)).reshape(-1).numpy()
    labels = (scores >= 0.5).astype(int)
    favorable = labels == desired_label
    return counterfactual, scores, affected_mask, successful, favorable, runtime


try:
    from raiutils.exceptions import UserConfigValidationException
except ImportError:
    UserConfigValidationException = ValueError


def distribution_metrics(factual, counterfactual, scores, targets, seed, sqrt_wd):
    set_seed(seed)
    swd = SlicedWassersteinDivergence(factual.shape[1], N_PROJECTIONS)
    wd = WassersteinDivergence()
    fact = torch.tensor(factual.values, dtype=torch.float32)
    cf = torch.tensor(counterfactual.values, dtype=torch.float32)
    score = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)
    target = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    swd_value = np.sqrt(swd.distance(cf, fact, TRIM_DELTA)[0].item())
    wd_squared = wd.distance(score, target, TRIM_DELTA)[0].item()
    return float(swd_value), float(np.sqrt(wd_squared) if sqrt_wd else wd_squared)


def constraint_metrics(config, factual_raw, cf_raw, mean, std):
    mean_feature = config["mean"][0]
    std_feature = config["std"][0]
    lhs, rhs, coefficient = config["lsc"]
    lhs_norm = (cf_raw[lhs] - mean[lhs]) / std[lhs]
    rhs_norm = (cf_raw[rhs] - mean[rhs]) / std[rhs]
    fsd_feature, direction = config["fsd"]
    fsd = FSDConstraintTorch.check_FSD(
        factual_raw[fsd_feature].to_numpy(), cf_raw[fsd_feature].to_numpy(), direction=direction
    )
    return {
        "Mean value": float(cf_raw[mean_feature].mean()),
        "Std value": float(cf_raw[std_feature].std(ddof=config["std_ddof"])),
        "LSC MSE": float(np.mean((lhs_norm - coefficient * rhs_norm) ** 2)),
        "FSD violations": int(len(fsd["violations"])),
    }


def domain_metrics(X_train_raw, factual_raw, cf_raw):
    changed = (cf_raw - factual_raw).abs().max(axis=1) > 1e-8
    below = (cf_raw < X_train_raw.min()).any(axis=1)
    above = (cf_raw > X_train_raw.max()).any(axis=1)
    integer_columns = [c for c in cf_raw if np.allclose(X_train_raw[c], np.round(X_train_raw[c]))]
    noninteger = (np.abs(cf_raw[integer_columns] - np.round(cf_raw[integer_columns])) > 1e-8).any(axis=1)
    invalid = changed & (below | above | noninteger)
    return {
        "Changed count": int(changed.sum()),
        "Invalid changed count": int(invalid.sum()),
        "Valid changed rate": float(1 - invalid.sum() / max(changed.sum(), 1)),
        "Out-of-range count": int((changed & (below | above)).sum()),
        "Non-integer count": int((changed & noninteger).sum()),
    }


def run_dataset(name, root, seeds):
    config = dataset_config(name, root)
    X_train, y_train, factual_norm, factual_raw, mean, std = prepare_named_dataset(name, root)
    model = train_model(X_train, y_train)
    dice = build_dice(X_train, y_train, model)
    rows = []
    populations = {}
    for seed in seeds:
        set_seed(TRAIN_SEED)
        a, b = config["beta"]
        targets = torch.distributions.beta.Beta(a, b).sample((SAMPLE_SIZE,)).numpy()
        cf_norm, scores, affected, successful, favorable, runtime = generate_population(
            dice, model, factual_norm, config["desired_label"], seed
        )
        cf_raw = cf_norm * std + mean
        swd, wd = distribution_metrics(
            factual_norm, cf_norm, scores, targets, seed, config["sqrt_wd"]
        )
        row = {
            "Dataset": name,
            "Method": "DiCE",
            "seed": seed,
            "Affected count": int(affected.sum()),
            "Successful recourse": int(successful[affected].sum()),
            "Coverage": float(successful[affected].mean()) if affected.any() else 1.0,
            "Favorable rate": float(favorable.mean()),
            "SWD": swd,
            "WD": wd,
            "Runtime (s)": runtime,
        }
        row.update(constraint_metrics(config, factual_raw.reset_index(drop=True), cf_raw.reset_index(drop=True), mean, std))
        row.update(domain_metrics(X_train * std + mean, factual_raw.reset_index(drop=True), cf_raw.reset_index(drop=True)))
        rows.append(row)
        populations[seed] = cf_raw.assign(model_score=scores, favorable=favorable)
    return pd.DataFrame(rows), populations


def summarize(frame):
    numeric = [c for c in frame.select_dtypes(include=np.number) if c != "seed"]
    result = frame.groupby(["Dataset", "Method"])[numeric].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def save_results(runs, populations, output_dir, prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    runs.to_csv(output_dir / f"{prefix}_runs.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    with pd.ExcelWriter(output_dir / f"{prefix}_audit.xlsx") as writer:
        runs.to_excel(writer, sheet_name="runs", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        for dataset, seed_frames in populations.items():
            for seed, frame in seed_frames.items():
                frame.to_excel(writer, sheet_name=f"{dataset[:6]}_{seed}", index=False)
    return summary


def run_german(root, output_dir):
    runs, populations = run_dataset("German Credit", root, [TRAIN_SEED])
    summary = save_results(runs, {"German": populations}, output_dir, "german_credit_dice")
    return runs, summary


def run_other_datasets(root, output_dir):
    all_runs = []
    populations = {}
    for name in ["Cardio", "HELOC", "Market"]:
        runs, generated = run_dataset(name, root, SEEDS)
        all_runs.append(runs)
        populations[name] = generated
    runs = pd.concat(all_runs, ignore_index=True)
    summary = save_results(runs, populations, output_dir, "multi_dataset_dice_5runs")
    return runs, summary
