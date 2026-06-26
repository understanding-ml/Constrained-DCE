# Constrained Distributional Counterfactual Explanations

This repository contains the implementation of **Constrained Distributional Counterfactual Explanations (Constrained DCE)**.

The project extends the original Distributional Counterfactual Explanations (DCE) framework by adding a constraint management module. The goal is to generate distribution-level counterfactual explanations while also satisfying practical feature constraints, such as mean constraints, standard deviation constraints, linear structural constraints, and first-order stochastic dominance constraints.

## 1. Core Modules

| File Path                 | Description                                                                                                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `explainers/dce_v2.py`    | Implements the `DCEWithConstraints` class. It extends the original DCE framework with constraint penalty integration and GPU support.                                                                  |
| `explainers/manager.py`   | Implements the `ConstraintManager` class. It manages different constraint types and combines their penalty terms during optimization.                                                                  |
| `explainers/constraints/` | Contains the implementation of different constraint types, including mean, standard deviation, linear structural constraint, and first-order stochastic dominance.                                     |
| `explainers/recovery/`    | Contains post-processing recovery modules. `RecoveryManager` controls the recovery process, and the folder includes recovery methods for moment-based constraints such as mean and standard deviation. |
| `models/`                 | Contains predictive models used in the experiments, including MLP, logistic regression, SVM, and RBF-based models.                                                                                     |
| `utils/`                  | Contains utility functions for data processing, transformation, visualization, and logging.                                                                                                            |

## 2. Experimental Resources

| Path                                 | Description                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `data/`                              | Stores raw datasets used in the experiments.                                                                 |
| `germancredit/`                      | Stores German Credit case-study results and constraint experiment outputs.                                   |
| `HELOC/`                             | Contains experiment scripts and result files for the HELOC validation dataset.                               |
| `cardio/`                            | Contains experiment scripts and result files for the Cardio validation dataset.                              |
| `market/`                            | Contains experiment scripts and result files for the Marketing Campaign validation dataset.                  |
| `26_german_credit.ipynb`             | Full workflow notebook for the German Credit case study, including experiment execution and result analysis. |
| `HELOC_analysis.ipynb`               | Notebook for analyzing HELOC experiment results.                                                             |
| `cardio_analysis.ipynb`              | Notebook for analyzing Cardio experiment results.                                                            |
| `market_analysis.ipynb`              | Notebook for analyzing Marketing Campaign experiment results.                                                |
| `scripts/run_validation_datasets.sh` | Shell script for running validation dataset experiments in batch.                                            |

## 3. Installation

Install the required dependencies with:

```bash
pip install -r requirement.txt
```

## 4. Running Experiments

To run the German Credit case study, open and run:

```text
26_german_credit.ipynb
```

To run validation dataset experiments, use:

```bash
bash scripts/run_validation_datasets.sh
```

Individual dataset scripts can also be run directly, for example:

```bash
python cardio/cardio.py
python HELOC/HELOC.py
python market/market.py
```

## 5. Project Structure

```text
.
├── data/
├── explainers/
│   ├── constraints/
│   ├── recovery/
│   ├── dce.py
│   ├── dce_v2.py
│   └── manager.py
├── models/
├── utils/
├── germancredit/
├── HELOC/
├── cardio/
├── market/
├── experiments/
├── scripts/
└── 26_german_credit.ipynb
```

## 6. Notes

The main implementation is built around `DCEWithConstraints` and `ConstraintManager`. The constraint manager allows multiple constraint penalties to be added to the original DCE objective, while the recovery module provides optional post-processing for stricter feasibility under moment-based constraints.
