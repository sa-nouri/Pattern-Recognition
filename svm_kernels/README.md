# Support Vector Machine & Kernels

This project implements Support Vector Machines (SVM) with different kernel functions for classification tasks.

## Project Structure
```
Support Vector Machine & Kernels/
├── src/
│   ├── Poly_Linear_Kernels.py  # Implementation of polynomial and linear kernels
│   └── RBF.py                  # Implementation of RBF kernel
├── data/                       # Data directory
├── notebooks/                  # Jupyter notebooks for analysis
└── results/                    # Output directory for results
```

## Requirements
- Python 3.7+
- Dependencies listed in root requirements.txt

## Usage
1. Linear and Polynomial Kernels:
   ```python
   from src.Poly_Linear_Kernels import SVM
   model = SVM(kernel='linear')  # or kernel='poly'
   model.train(X, y)
   ```

2. RBF Kernel:
   ```python
   from src.RBF import RBF_SVM
   model = RBF_SVM(gamma=0.1)
   model.train(X, y)
   ```

## Results
The project demonstrates the performance of different kernel functions in SVM classification. Visualizations and analysis can be found in the notebooks directory.

## References
- See Problem.pdf for detailed problem description and requirements 
