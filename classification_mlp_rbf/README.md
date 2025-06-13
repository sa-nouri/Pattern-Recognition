# Classification Using MLP & RBF

This project implements and compares Multi-Layer Perceptron (MLP) and Radial Basis Function (RBF) networks for classification tasks.

## Project Structure
```
Classification Using MLP & RBF/
├── src/
│   ├── neural_net.py      # MLP implementation
│   ├── rbf.py            # RBF network implementation
│   └── dataloader.py     # Data loading utilities
├── data/
│   ├── data.csv          # Input features
│   └── labels.csv        # Target labels
├── notebooks/
│   └── mlp.ipynb         # Jupyter notebook with analysis
└── results/              # Output directory for results
```

## Requirements
- Python 3.7+
- Dependencies listed in root requirements.txt

## Usage
1. Data Preparation:
   ```python
   from src.dataloader import load_data
   X, y = load_data('data/data.csv', 'data/labels.csv')
   ```

2. MLP Training:
   ```python
   from src.neural_net import MLP
   model = MLP(input_size=2, hidden_size=10, output_size=1)
   model.train(X, y, epochs=100)
   ```

3. RBF Training:
   ```python
   from src.rbf import RBFNetwork
   model = RBFNetwork(input_size=2, num_centers=10, output_size=1)
   model.train(X, y)
   ```

## Results
The project compares the performance of MLP and RBF networks on the given dataset. Key findings and visualizations can be found in the notebooks directory.

## References
- See Problem.pdf for detailed problem description and requirements 
