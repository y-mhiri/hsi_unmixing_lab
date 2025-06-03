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

## 📊 Evaluation Metrics

- **Spectral Angle Mapper (SAM)**: Measures spectral similarity
- **Root Mean Square Error (RMSE)**: Reconstruction accuracy
- **Structural Similarity Index (SSIM)**: Perceptual image quality
- **Abundance constraints**: Non-negativity and sum-to-one validation

## 🧪 Example Scripts

### Basic Usage (`examples/basic_usage.py`)

```python
#!/usr/bin/env python3
"""
Basic usage example for the hyperspectral unmixing lab.
"""

import numpy as np
from src.data_loader import HyperspectralDataLoader, create_synthetic_data
from src.optimization.optimization_methods import HyperspectralUnmixer
from src.evaluation.metrics import UnmixingEvaluator

def main():
    # Test with synthetic data first
    print("Testing with synthetic data...")
    hsi_data, S_true, A_true = create_synthetic_data(
        height=50, width=50, bands=100, num_endmembers=3)
    
    Y = hsi_data.reshape(-1, hsi_data.shape[2]).T
    
    # Initialize unmixer
    unmixer = HyperspectralUnmixer()
    
    # Test different methods
    methods = {
        'Unconstrained': lambda: unmixer.unconstrained_least_squares(S_true, Y),
        'Simplex': lambda: unmixer.fully_constrained_least_squares(S_true, Y)[0]
    }
    
    evaluator = UnmixingEvaluator()
    
    for name, method in methods.items():
        A_estimated = method()
        aad = evaluator.abundance_angle_distance(A_true, A_estimated)
        print(f"{name}: AAD = {np.degrees(aad):.2f}°")

if __name__ == "__main__":
    main()
```

### Advanced Example (`examples/advanced_examples.py`)

```python
#!/usr/bin/env python3
"""
Advanced examples including blind unmixing and custom algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import HyperspectralDataLoader
from src.optimization.optimization_methods import HyperspectralUnmixer, BlindUnmixer
from src.evaluation.metrics import UnmixingEvaluator, compute_endmember_similarity

def compare_initialization_strategies():
    """Compare different initialization strategies for blind unmixing."""
    
    # Load real data
    loader = HyperspectralDataLoader("data/indian_pines/")
    hsi_data, _ = loader.load_indian_pines()
    Y, _ = loader.vectorize_data()
    
    # Test blind unmixing with different initializations
    unmixer = HyperspectralUnmixer()
    blind_unmixer = BlindUnmixer(unmixer)
    
    strategies = ['random']  # Could add 'vca', 'nfindr', etc.
    results = {}
    
    for strategy in strategies:
        print(f"Testing {strategy} initialization...")
        S_est, A_est, obj_vals = blind_unmixer.block_coordinate_descent(
            Y, num_endmembers=3, max_iter=20, initialization=strategy)
        
        # Evaluate convergence
        final_objective = obj_vals[-1]
        convergence_rate = (obj_vals[0] - obj_vals[-1]) / obj_vals[0]
        
        results[strategy] = {
            'final_objective': final_objective,
            'convergence_rate': convergence_rate,
            'endmembers': S_est,
            'abundances': A_est
        }
        
        print(f"  Final objective: {final_objective:.6f}")
        print(f"  Convergence rate: {convergence_rate:.4f}")
    
    return results

def custom_constraints_example():
    """Example of implementing custom constraints."""
    
    class CustomUnmixer(HyperspectralUnmixer):
        def volume_constrained_unmixing(self, S, Y, max_volume=1.0):
            """
            Example of custom constraint: limit abundance volume.
            This is just a demonstration - not a standard method.
            """
            # Start with simplex solution
            A, _ = self.fully_constrained_least_squares(S, Y)
            
            # Apply volume constraint (simplified example)
            for p in range(A.shape[1]):
                abundance_volume = np.prod(A[:, p])
                if abundance_volume > max_volume:
                    # Simple scaling (not optimal, just for demonstration)
                    scale_factor = (max_volume / abundance_volume) ** (1/A.shape[0])
                    A[:, p] *= scale_factor
                    # Re-normalize to sum to 1
                    A[:, p] /= np.sum(A[:, p])
            
            return A
    
    # Test custom unmixer
    from src.data_loader import create_synthetic_data
    
    hsi_data, S_true, A_true = create_synthetic_data(
        height=20, width=20, bands=50, num_endmembers=3)
    Y = hsi_data.reshape(-1, hsi_data.shape[2]).T
    
    custom_unmixer = CustomUnmixer()
    A_custom = custom_unmixer.volume_constrained_unmixing(S_true, Y)
    
    print("Custom constraint applied successfully!")
    print(f"Volume constraint satisfaction: {np.all(np.prod(A_custom, axis=0) <= 1.0)}")

if __name__ == "__main__":
    print("Running advanced examples...")
    
    # Example 1: Initialization comparison
    try:
        init_results = compare_initialization_strategies()
        print("✓ Initialization comparison completed")
    except Exception as e:
        print(f"✗ Initialization comparison failed: {e}")
    
    # Example 2: Custom constraints
    try:
        custom_constraints_example()
        print("✓ Custom constraints example completed")
    except Exception as e:
        print(f"✗ Custom constraints example failed: {e}")
```

## 🛠️ Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/
```

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

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add convergence monitoring
def debug_convergence(obj_values, tolerance=1e-6):
    for i, val in enumerate(obj_values[1:], 1):
        rel_change = abs(val - obj_values[i-1]) / obj_values[i-1]
        print(f"Iter {i}: Obj={val:.6f}, Rel.Change={rel_change:.2e}")
        if rel_change < tolerance:
            print(f"Converged at iteration {i}")
            break
```

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

