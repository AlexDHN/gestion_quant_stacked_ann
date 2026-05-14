# Forecasting Volatility with a Stacked Model Based on a Hybridized ANN

> Academic replication project — Université Paris-Dauphine  
> Based on the paper by **Ramos-Pérez, Alonso-González & Núñez-Velázquez (2021)**

---

## Overview

This project implements and replicates the methodology described in:

> *Ramos-Pérez, E., Alonso-González, P. J., & Núñez-Velázquez, J. J. (2021). Forecasting volatility with a stacked model based on a hybridized Artificial Neural Network. Expert Systems with Applications.*

The core idea is to combine multiple volatility models — including classical econometric specifications (GARCH family) and the Heston stochastic volatility model — into a **stacked ensemble** driven by a hybridized Artificial Neural Network (ANN). The goal is to improve out-of-sample volatility forecasting accuracy beyond what any single model achieves individually.

---

## Authors

**Gestion Quantitative, Université Paris-Dauphine**

- Alex Dhenin
- Lefevre
- Manelli
- Nitcheu

---

## Repository Structure

```
.
├── classes/                    # Core Python classes (model definitions, stacking logic)
├── data/                       # Financial time series data
├── fig/                        # Generated figures and plots
├── models/
│   └── torch/                  # Saved PyTorch model weights
├── output/                     # Numerical results and evaluation metrics
├── utils/                      # Helper functions (data loading, metrics, preprocessing)
│
├── main.ipynb                          # Main notebook: full pipeline replicating paper statistics
├── training.ipynb                      # Training procedure for the stacked ANN
├── training_prop_models.ipynb          # Training of individual (base) models
├── comparaison_models.ipynb            # Comparison of all models on evaluation metrics
├── bootstraping.ipynb                  # Bootstrap-based confidence intervals
├── heston.ipynb                        # Heston stochastic volatility model implementation
├── ann_example.ipynb                   # Standalone ANN example and architecture walkthrough
├── best_init_function.ipynb            # Weight initialization experiments
│
├── rapport_DHENIN_LEFEVRE_MANELLI_NITCHEU.pdf   # Full academic report (French)
├── presentation_G5.pptx                          # Presentation slides
└── 21_Ramos-Pérez et alii_Forecasting volatility with a stacked model based on a hybridized ANN.pdf
```

---

## Methodology

The replication follows a three-stage pipeline:

**1. Base models (level-0 learners)**  
Several volatility models are trained independently to produce forecasts used as inputs to the meta-learner:
- GARCH / EGARCH / GJR-GARCH
- Heston stochastic volatility model
- Standalone ANN

**2. Stacking (meta-learning)**  
A hybridized ANN acts as the meta-learner, combining the base model forecasts. Weight initialization is carefully tuned (see `best_init_function.ipynb`) to improve convergence.

**3. Evaluation**  
Models are compared on standard forecasting metrics (MSE, QLIKE, etc.) using bootstrap resampling to assess statistical significance of performance differences.

---

## Getting Started

### Prerequisites

Python 3.8+ is recommended. Install the required dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn scipy arch
```

> Depending on your environment, additional packages may be required. Check the import cells at the top of each notebook.

### Running the Project

The notebooks are designed to be executed in the following order:

1. **`training_prop_models.ipynb`** — Train and save the individual base models
2. **`heston.ipynb`** — Calibrate the Heston model and generate volatility estimates
3. **`training.ipynb`** — Train the stacked ANN meta-learner
4. **`main.ipynb`** — Reproduce the paper's main statistics and results
5. **`comparaison_models.ipynb`** — Compare all models and generate performance tables
6. **`bootstraping.ipynb`** — Run bootstrap tests for statistical significance

`ann_example.ipynb` and `best_init_function.ipynb` are standalone exploratory notebooks and can be run independently.

---

## Key Results

The stacked ANN model achieves lower forecasting error than any individual base model across the evaluation metrics reported in the original paper, consistent with the findings of Ramos-Pérez et al. (2021). Detailed results, figures, and statistical tests are available in the `output/` and `fig/` directories, and are discussed in full in the [academic report](rapport_DHENIN_LEFEVRE_MANELLI_NITCHEU.pdf).

---

## Reference

```bibtex
@article{ramosperez2021,
  title     = {Forecasting volatility with a stacked model based on a hybridized Artificial Neural Network},
  author    = {Ramos-Pérez, Eduardo and Alonso-González, Pablo J. and Núñez-Velázquez, José Javier},
  journal   = {Expert Systems with Applications},
  year      = {2021}
}
```
