# Clustering Algorithms

This project implements various clustering algorithms including K-Means, Agglomerative Clustering, and Affinity Propagation.

## Project Structure
```
Clustering/
├── src/
│   ├── K_Means.py              # K-Means clustering implementation
│   ├── Agglomerative_Clustering.py  # Hierarchical clustering implementation
│   └── Affinity_Propagation.py # Affinity Propagation implementation
├── data/                       # Data directory
├── notebooks/                  # Jupyter notebooks for analysis
└── results/                    # Output directory for results
```

## Requirements
- Python 3.7+
- Dependencies listed in root requirements.txt

## Usage
1. K-Means Clustering:
   ```python
   from src.K_Means import KMeans
   model = KMeans(n_clusters=3)
   labels = model.fit_predict(X)
   ```

2. Agglomerative Clustering:
   ```python
   from src.Agglomerative_Clustering import AgglomerativeClustering
   model = AgglomerativeClustering(n_clusters=3)
   labels = model.fit_predict(X)
   ```

3. Affinity Propagation:
   ```python
   from src.Affinity_Propagation import AffinityPropagation
   model = AffinityPropagation()
   labels = model.fit_predict(X)
   ```

## Results
The project compares different clustering algorithms and their performance on various datasets. Visualizations and analysis can be found in the notebooks directory.

## References
- See Problem.pdf for detailed problem description and requirements 
