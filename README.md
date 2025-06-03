# Remote Sensing Optimization Lab

This repository contains the complete implementation for the hyperspectral unmixing optimization lab. Students will learn to solve inverse problems using various constrained optimization techniques.

## 🎯 Learning Objectives

- Derive data models and objective functions for inverse problems
- Implement descent algorithms for multi-objective optimization 
- Handle constrained optimization using Lagrange multipliers and projections
- Benchmark optimization algorithms and measure performance
- Work with real hyperspectral remote sensing data
- Understand the relationship between physical constraints and mathematical optimization

## 📁 Repository Structure

```
remote_sensing_lab/
├── src/                        # Source code modules
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── visualization.py         # Plotting utilities
│   ├── optimization.py          # Optimization algorithms
│   ├── metric.py                # Performance metrics
│   └── utils/                   # Helper functions
├── notebooks/                  # Jupyter notebook exercises
├── data/                       # Data directory
├── examples/                   # Example scripts
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/remote-sensing-lab.git
cd remote-sensing-lab

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Download Data

Download the Indian Pines dataset:
- Visit: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- Download `Indian_pines_corrected.mat` and `Indian_pines_gt.mat`
- Place in `data/indian_pines/` directory

### 3. Run Basic Example

```python
from src.data_loader import HyperspectralDataLoader
from src.optimization.optimization_methods import HyperspectralUnmixer
from src.visualization import HSIVisualizer

# Load data
loader = HyperspectralDataLoader("data/indian_pines/")
hsi_data, ground_truth = loader.load_indian_pines()

# Extract endmembers and vectorize data
S, names = loader.extract_class_spectra([1, 2, 3])
Y, _ = loader.vectorize_data()

# Solve unmixing problem
unmixer = HyperspectralUnmixer()
A = unmixer.fully_constrained_least_squares(S, Y)

# Visualize results
visualizer = HSIVisualizer()
visualizer.plot_abundance_maps(A, hsi_data.shape[:2], names)
```

## 📚 Lab Sections

### Section I: Data Exploration
- Load and visualize hyperspectral data
- Understand the Indian Pines dataset
- Extract endmember spectra from ground truth

### Section II: Unconstrained Optimization
- Implement analytical least squares solution
- Analyze problems with unconstrained abundances
- Performance evaluation metrics

### Section III: Constrained Optimization
- Sum-to-one constraints using Lagrange multipliers
- Non-negativity constraints with projected gradient
- Full simplex constraints
- Compare all methods

### Section IV: Blind Unmixing (Advanced)
- Block Coordinate Descent algorithm
- Joint estimation of endmembers and abundances
- Initialization strategies and convergence analysis

## 🔬 Key Algorithms Implemented

1. **Unconstrained Least Squares**: `A* = (S^T S)^(-1) S^T Y`
2. **Sum-to-One Constrained**: Analytical solution using Lagrange multipliers
3. **Projected Gradient Descent**: For non-negativity and simplex constraints
4. **Simplex Projection**: Duchi et al. algorithm for unit simplex
5. **Block Coordinate Descent**: For blind unmixing

## 📈 Performance Tips

1. **Step Size Selection**: Use adaptive step sizes for better convergence
2. **Initialization**: Good initialization significantly improves blind unmixing
3. **Convergence Criteria**: Monitor relative changes in objective function
4. **Memory Management**: Process large datasets in batches if needed

## 🔍 Troubleshooting

### Common Issues

1. **Singular Matrix Error**: Occurs when endmembers are linearly dependent
   - Solution: Use pseudo-inverse or add regularization

2. **Slow Convergence**: Projected gradient descent converging slowly
   - Solution: Reduce step size or use adaptive step size

3. **Memory Issues**: Large hyperspectral datasets
   - Solution: Process subsets or use batch processing

4. **Negative Abundances**: Unconstrained solution gives negative values
   - Solution: Use constrained optimization methods

## 📚 References

1. Bioucas-Dias, J. M., et al. "Hyperspectral unmixing overview: Geometrical, statistical, and sparse regression-based approaches." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 5.2 (2012): 354-379.

2. Duchi, J., et al. "Efficient projections onto the l1-ball for learning in high dimensions." Proceedings of the 25th international conference on Machine learning. 2008.

3. Heinz, D. C. (2001). Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery. IEEE transactions on geoscience and remote sensing, 39(3), 529-545.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -am 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Create a Pull Request

