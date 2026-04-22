# Deep Ranking with Heterogeneous Effects

This repository contains the implementation and reproducibility workflow for the paper
[Deep Ranking with Heterogeneous Effect](https://arxiv.org/abs/2604.16129).

1. Source code for simulation and predicting ATP tennis match outcomes.
2. The collected dataset is stored in `real_data/data/datasets_processed/` and contains ATP tennis data (ATP 250, 500, 1000, Finals, and Grand Slams) between 2016 and 2025.

All commands below assume you start from the repository root.

## Repository Structure

```text
.
├── simulation/                  # Simulation experiments
│   ├── generator.py             # Ranking data generator with heterogeneous covariate effects
│   ├── algorithm.py             # PL and PlusDC algorithm
│   ├── deep_algorithm.py        # Deep ranking estimation routines
│   ├── simulation.py            # Training process
│   ├── Figure_3.py              # Simulation visualization script
│   └── run.sh                   # Full simulation script
├── real_data/                   # ATP tennis match prediction experiments
│   ├── main_train_realdata.py   # Model training for ATP data
│   ├── main_optimal_metrics.py  # Metric aggregation and model selection summaries
│   ├── fig6_radar_plot_enhanced.py
│   ├── fig7_visualize_single_model.py
│   ├── run_train.sh             # Training script
│   ├── run_full_experiment.sh   # End-to-end real-data workflow
│   ├── packages/                # Algorithms for real-data experiments
│   └── data/                    # Processed ATP datasets
├── requirements.txt
└── README.md
```

## Part 1: Environment Setup

The experiments were developed with Python 3.9. Create a clean environment and
install the dependencies from the repository root:

```bash
conda create -n deeprank python=3.9
conda activate deeprank
pip install -r requirements.txt
```

## Part 2: Full Experiment Pipeline for simulation

To run the full workflow:

```bash
cd simulation
bash run.sh
```

## Part 3: Full Experiment Pipeline for ATP Tennis Match Prediction

To run the full workflow:

```bash
cd real_data
bash run_full_experiment.sh
```


## Part 4: ATP Tennis Match Prediction Example Commands

The commands below show an example run for ATP Tennis Match Prediction with a single random seed / replication.
Run these commands from the `real_data` folder:

```bash
cd real_data
```

#### Feature Groups

- **MI** (17 dim): Match Information
- **PP** (15 dim): Player Profile
- **TS** (34 dim): Recent Technical Statistics
- **Total**: MI_PP_TS = 66 dimensions

Processed datasets are stored in `real_data/data/datasets_processed/`. Each
folder corresponds to one `--feature_name` option:

- `MI_dim17`: match context only.
- `PP_dim15`: player profile only.
- `TS_dim34`: recent technical statistics performance statistics only.
- `MI_PP_dim32`, `MI_TS_dim51`, `PP_TS_dim49`: pairwise feature combinations.
- `MI_PP_TS_dim66`: all feature groups combined.

---
### 4.1 Train Models

```bash
python main_train_realdata.py --feature_name MI_PP_TS_dim66 --sim_id 1 --hidden_dim 16 --hidden_num 3 --bs 32 --lr 0.001 --weight_decay 0.0001

```

Train Deep, Deep_no_u, PL, PlusDC models

we can also use a bash script to train all models across multiple feature sets (and optionally reps/seeds).

```bash
# Single feature groups
bash run_train.sh
```

> **Tip:** You can increase training speed by running multiple training jobs in parallel. Edit `run_train.sh` and increase `CONCURRENT` to the number of processes you want to run at the same time. A larger value can finish the grid search faster, but it will also use more CPU resources.
>
> ```bash
> # Number of concurrent processes
> CONCURRENT=1
> ```

---

### 4.2 Calculate Optimal Metrics

Example with one replication:

```bash
python main_optimal_metrics.py --rep_start 1 --rep_end 1
```

#### Evaluation Metrics

- Likelihood (train/test/val)
- Win Rate (train/test/val)
- Brier Score (train/test/val)

Evaluates performance and outputs `metrics_summary.json`, `metrics_summary.csv`, `accumulate_dim.png`

---

### 4.3 Visualize Results

Example visualization for one replication:

```bash
python fig6_radar_plot_enhanced.py --style style2
python fig7_visualize_single_model.py --feature_name MI_PP_TS_dim66 --model_type Deep --rep_id 1 --num_reps 1
```

Generates trajectories plots and radar charts

---


## Citation

If you use this code or the collected dataset, please cite the corresponding paper:

```bibtex
@article{luo2026deep,
  title={Deep Ranking with Heterogeneous Effect},
  author={Luo, Yuanhang and Fang, Shuxing and Han, Ruijian and Xu, Yiming},
  journal={arXiv preprint arXiv:2604.16129},
  year={2026}
}
```
