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
git clone https://github.com/understanding-ml/Constrained-DCE.git
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
