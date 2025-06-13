# Feature Conditioning

This project implements various feature conditioning and selection techniques including Linear Discriminant Analysis (LDA) and Forward Selection.

## Project Structure
```
Feature Conditioning/
├── src/
│   ├── LDA.py                 # Linear Discriminant Analysis implementation
│   └── Forward_Selection.py   # Forward feature selection implementation
├── data/                      # Data directory
├── notebooks/                 # Jupyter notebooks for analysis
└── results/                   # Output directory for results
```

## Requirements
- Python 3.7+
- Dependencies listed in root requirements.txt

## Usage
1. Linear Discriminant Analysis:
   ```python
   from src.LDA import LDA
   model = LDA(n_components=2)
   X_transformed = model.fit_transform(X, y)
   ```

2. Forward Selection:
   ```python
   from src.Forward_Selection import ForwardSelection
   selector = ForwardSelection(n_features=5)
   X_selected = selector.fit_transform(X, y)
   ```

## Results
The project demonstrates different feature conditioning and selection techniques. Visualizations and analysis can be found in the notebooks directory.

## References
- See Problem.pdf for detailed problem description and requirements 
