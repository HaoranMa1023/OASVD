# OASVD: Online Adaptive SVD Experiments

This repository contains the reference implementation and experimental notebooks for the **Online Adaptive SVD (OASVD)** framework.  OASVD is a streaming low‑rank approximation method that treats the matrix approximation problem as a **feedback control loop**: at each time step it maintains an approximate singular value decomposition and adapts the truncation rank based on online estimates of residual spectral energy, approximation error and orthogonality, all under a fixed per‑step computational budget.

The primary goal of this repository is to make the algorithm and its numerical experiments easy to reproduce, study and extend.  It includes a Python package under `oasvd/` implementing the core algorithms and baselines, a set of Colab‑ready notebooks under `notebooks/` for each of the experiments reported in our paper, and placeholders for data and results.

## Contents

The top‑level layout of this repository is as follows:

```
OASVD/
├── LICENSE          # MIT License (already generated)
├── README.md        # This file
├── requirements.txt # Python dependencies
├── .gitignore       # Files/directories to exclude from version control
│
├── oasvd/           # Core library code implementing OASVD and baselines
│   ├── __init__.py
│   ├── incremental_svd.py    # Incremental SVD updates
│   ├── spectral_probe.py     # Residual spectral probing utilities
│   ├── control_law.py        # OASVD control loop and rank adaptation
│   ├── baselines.py          # Fixed‑rank and full SVD baselines
│   ├── metrics.py            # Error/jitter/coverage metrics
│   ├── plotting.py           # Common plotting utilities
│   └── utils.py              # Miscellaneous helpers (seeding, timers)
│
├── notebooks/       # Jupyter/Colab notebooks for each experiment
│   ├── exp1_synthetic.ipynb  # Piecewise‑spectral synthetic data with shocks
│   ├── exp2_heat.ipynb       # Heat equation snapshots
│   ├── exp3_cylinder.ipynb   # 2D cylinder flow snapshots
│   ├── exp4_kernel.ipynb     # Parametric kernel matrices
│   └── exp5_ablation.ipynb   # Ablation study
│
├── data/            # External or generated data (large files are gitignored)
│   └── README.md    # Instructions for obtaining data files
│
└── results/         # Generated figures and CSV logs (gitignored)
    └── README.md    # Description of result organisation
```

## Installation

This codebase requires Python 3.9 or later.  To install the dependencies, clone the repository and run:

```bash
git clone https://github.com/HaoranMa1023/OASVD.git
cd OASVD
pip install -r requirements.txt
pip install -e .
```

The editable install (`pip install -e .`) makes the `oasvd` package available on your `PYTHONPATH` so that the notebooks can import the core algorithms.

### Running in Google Colab

Each notebook in the `notebooks/` directory is designed to run end‑to‑end in Google Colab.  A typical workflow is:

1. Upload this repository (or clone via Git) into your Google Drive.
2. Open a notebook (e.g. `notebooks/exp1_synthetic.ipynb`) in Colab.
3. In the first cell, install the package and its dependencies:

   ```python
   !pip install -r /content/drive/MyDrive/OASVD/requirements.txt
   !pip install -e /content/drive/MyDrive/OASVD
   ```

4. Run the remaining cells.  Each notebook saves its metrics and plots into the `results/` directory.

## Experiments

The repository contains notebooks for five distinct experiments, corresponding to the study described in our accompanying paper:

1. **Synthetic Piecewise‑Spectral Streams (`exp1_synthetic.ipynb`)** – Generates matrix streams with piecewise spectral behaviour and injected rank‑1 shocks to test the algorithm’s responsiveness and rank adaptation.
2. **Heat Equation Snapshots (`exp2_heat.ipynb`)** – Constructs snapshot matrices from 1D/2D heat equation simulations to explore error control and rank modulation in smooth, slowly varying spectra.
3. **Cylinder Flow Snapshots (`exp3_cylinder.ipynb`)** – Uses 2D cylinder flow data exhibiting strong non‑stationarity and vortex shedding to stress test OASVD’s ability to track rapid spectral changes while maintaining numerical stability.
4. **Parametric Kernel Matrices (`exp4_kernel.ipynb`)** – Forms kernel matrices along a parameter trajectory (e.g. varying RBF bandwidth) to evaluate how well OASVD can reuse subspaces across parameters.
5. **Ablation Study (`exp5_ablation.ipynb`)** – Systematically disables components of OASVD (spectral probing, hysteresis, error feedback, re‑orthogonalisation) to quantify their individual contributions.

Each notebook sets up the data, runs OASVD and baseline methods, computes metrics such as relative error, rank jitter, coverage and orthogonality, and generates the figures used in our paper.  The notebooks are well annotated and can serve both as a reproduction of our results and as a starting point for new experiments.

## Data

Large raw datasets (e.g. CFD snapshots) are **not** committed to this repository.  See `data/README.md` for instructions on obtaining or generating the necessary data files for each experiment.  Synthetic data and small toy examples are generated on the fly by the notebooks.

## Contributing

Contributions in the form of bug reports, feature requests or pull requests are welcome.  If you design new experiments or improve the algorithms, please consider adding a notebook under `notebooks/` and updating this README accordingly.

## Citation

If you use this code or the OASVD algorithm in your own work, please cite our paper.  A machine‑readable citation file (`CITATION.cff`) is provided in the repository.
