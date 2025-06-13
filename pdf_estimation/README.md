# Probability Density Function (PDF) Estimation

This project implements both parametric and non-parametric methods for estimating probability density functions.

## Project Structure
```
PDF Estimation/
├── Parametric PDF Estimation/
│   ├── src/                    # Source code for parametric methods
│   ├── data/                   # Data directory
│   ├── notebooks/             # Jupyter notebooks for analysis
│   └── results/               # Output directory for results
└── Non-Parametric PDF Estimation/
    ├── src/                    # Source code for non-parametric methods
    ├── data/                   # Data directory
    ├── notebooks/             # Jupyter notebooks for analysis
    └── results/               # Output directory for results
```

## Requirements
- Python 3.7+
- Dependencies listed in root requirements.txt

## Usage
1. Parametric PDF Estimation:
   ```python
   from src.parametric import GaussianPDF
   model = GaussianPDF()
   pdf = model.estimate(X)
   ```

2. Non-Parametric PDF Estimation:
   ```python
   from src.nonparametric import KernelDensity
   model = KernelDensity(bandwidth=0.5)
   pdf = model.estimate(X)
   ```

## Results
The project compares different PDF estimation methods and their performance on various datasets. Visualizations and analysis can be found in the notebooks directory.

## References
- See Problem.pdf for detailed problem description and requirements 
