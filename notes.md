# Remote Sensing Optimization Lab - Repository Structure

```
remote_sensing_lab/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── indian_pines/
│   │   ├── Indian_pines_corrected.mat
│   │   ├── Indian_pines_gt.mat
│   │   └── README.md
│   └── sample_data/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── visualization.py         # Plotting and visualization utilities
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── unconstrained.py     # Unconstrained least squares
│   │   ├── constrained.py       # Sum-to-one and non-negativity constraints
│   │   ├── projected_gradient.py # Projected gradient algorithms
│   │   ├── simplex_projection.py # Simplex projection utilities
│   │   └── blind_unmixing.py    # Block coordinate descent
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # SAM, RMSE, SSIM evaluation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # General utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_unconstrained_unmixing.ipynb
│   ├── 03_constrained_unmixing.ipynb
│   ├── 04_projected_gradient.ipynb
│   ├── 05_blind_unmixing.ipynb
│   └── complete_lab_solution.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_optimization.py
│   └── test_evaluation.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_examples.py
└── docs/
    ├── lab_instructions.md
    └── api_reference.md
```

## Key Features

- **Modular Design**: Each optimization method is in its own module
- **Educational Structure**: Clear separation between concepts
- **Jupyter Notebooks**: Step-by-step guided exercises
- **Testing**: Unit tests for validation
- **Documentation**: API reference and examples
- **Data Management**: Organized data directory structure