# Constrained Distributional Counterfactual Explanations

This repository contains the implementation and experimental resources for
**Constrained Distributional Counterfactual Explanations (CDCE)**.

CDCE extends Distributional Counterfactual Explanations (DCE) by incorporating
explicit feasibility constraints into distribution-level counterfactual
optimization. The implementation supports four constraint families:

- mean constraints;
- standard-deviation constraints;
- linear structural constraints (LSC); and
- first-order stochastic dominance (FSD) constraints.

The repository also includes an optional recovery procedure for obtaining
strict moment feasibility and evaluation code for the DiCE and GLOBE-CE
baselines.

## Method overview

The original DCE objective searches for a generated population that remains
close to the factual population in input space while matching a desired output
distribution. CDCE augments this objective with constraint penalties:

```text
CDCE objective
  = input-distribution distance
  + output-distribution distance
  + weighted feasibility penalty.
```

`DCEWithConstraints` performs the optimization, while `ConstraintManager`
constructs and combines the selected constraint penalties. Moment constraints
can additionally use the recovery module when strict feasibility is required.

## Repository structure

```text
.
|-- explainers/
|   |-- constraints/
|   |   |-- MeanConstraint.py
|   |   |-- StdConstraint.py
|   |   |-- LSCConstraint.py
|   |   `-- FSDConstraint.py
|   |-- recovery/
|   |   |-- RecoveryManager.py
|   |   |-- MeanRecovery.py
|   |   `-- StdRecovery.py
|   |-- dce.py
|   |-- dce_v2.py
|   |-- distances.py
|   `-- manager.py
|-- baselines/
|   |-- dice/
|   `-- globe-ce/
|-- germancredit/
|-- HELOC/
|-- cardio/
|-- market/
|-- data/
|-- models/
|-- utils/
|-- experiments/
|-- scripts/
|-- 26_german_credit.ipynb
|-- HELOC_analysis.ipynb
|-- cardio_analysis.ipynb
`-- market_analysis.ipynb
```

### Core implementation

| Path | Description |
| --- | --- |
| `explainers/dce.py` | Original unconstrained DCE implementation. |
| `explainers/dce_v2.py` | CDCE implementation through the `DCEWithConstraints` class. |
| `explainers/manager.py` | Constraint configuration and penalty aggregation. |
| `explainers/constraints/` | Mean, standard-deviation, LSC, and FSD penalties. |
| `explainers/recovery/` | Optional recovery procedures for strict moment feasibility. |
| `explainers/distances.py` | Sliced Wasserstein and Wasserstein distance utilities. |
| `models/` | Prediction models used by the experiments. |

### Experiments and analysis

| Path | Description |
| --- | --- |
| `26_german_credit.ipynb` | German Credit case-study workflow and analysis. |
| `HELOC/HELOC.py` | HELOC experiments. |
| `cardio/cardio.py` | Cardio experiments. |
| `market/market.py` | Marketing Campaign experiments. |
| `HELOC_analysis.ipynb` | HELOC result analysis. |
| `cardio_analysis.ipynb` | Cardio result analysis. |
| `market_analysis.ipynb` | Marketing Campaign result analysis. |
| `baselines/dice/` | DiCE evaluation code, notebooks, and saved summaries. |
| `baselines/globe-ce/` | GLOBE-CE evaluation code, notebooks, and saved summaries. |

## Installation

Clone the repository and enter its root directory:

```bash
cd Constrained-DCE
```

Creating an isolated Python environment is recommended:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
pip install -r requirement.txt
```

The analysis notebooks and saved Excel outputs additionally require Jupyter
and `openpyxl`. The DiCE baseline requires `dice-ml`:

```bash
pip install jupyter openpyxl dice-ml
```

## Running the experiments

Run commands from the repository root so that local package and data paths are
resolved correctly.

### German Credit case study

Open and execute:

```text
26_german_credit.ipynb
```

The notebook contains the German Credit DCE/CDCE workflow, constraint
evaluation, and recovery analysis.

### Validation datasets

The three validation experiments can be run separately:

```bash
python HELOC/HELOC.py
python cardio/cardio.py
python market/market.py
```

Each experiment evaluates unconstrained DCE and the applicable CDCE constraint
families. The analysis notebooks read the saved experimental outputs and
compute the reported five-run summaries.

### Baselines

DiCE:

```text
baselines/dice/german_credit_dice_audit.ipynb
baselines/dice/multi_dataset_dice_5runs.ipynb
```

GLOBE-CE:

```text
baselines/globe-ce/german_credit_globe_ce_audit.ipynb
baselines/globe-ce/multi_dataset_globe_ce_5runs.ipynb
```

Saved per-run results and aggregate summaries are available under the
corresponding `results/` directories. The validation summaries use seeds
40--44.

## Datasets

The experiments cover four public binary-classification datasets:

| Dataset | Role |
| --- | --- |
| German Credit | Detailed institutional case study. |
| HELOC | Credit-domain validation. |
| Cardio | Health-domain validation. |
| Marketing Campaign | Business-domain validation. |

Dataset loading and preprocessing are implemented in the corresponding
experiment scripts. Review the terms and licenses of the original datasets
before redistribution or downstream use.

## Constraint configuration

Constraints are passed to `ConstraintManager` as configuration dictionaries.
A mean upper-bound constraint, for example, has the following form:

```python
configs = [
    {
        "type": "mean",
        "bounds": {"feature_name": target_value},
        "lambda": penalty_weight,
    }
]
```

The dataset scripts contain the exact feature choices, bounds, penalty
weights, optimization settings, and random seeds used for the reported
experiments.

## Reproducibility notes

- DCE and CDCE use the same processed data, explained population, prediction
  model, and target output within each dataset.
- The prediction model is held fixed during counterfactual optimization.
- The validation experiments report results over five runs with seeds 40--44.
- Tables in the main paper report five-run means; the complete mean and
  standard-deviation audit is produced by the analysis notebooks.
- DiCE and GLOBE-CE results are evaluated with the same constraint metrics and
  distributional-distance conventions used for DCE and CDCE.

## Responsible use

The supplied constraints represent experimental feasibility assumptions, not
universal policy recommendations. Appropriate constraints depend on the
application, affected population, data-generating process, and institutional
context. CDCE does not automatically guarantee individual-level actionability,
immutability, causal validity, or fairness; these requirements must be assessed
separately for any deployment.
