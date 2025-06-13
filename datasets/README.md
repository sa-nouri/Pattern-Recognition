# Datasets

This directory contains all the datasets used across different projects in this repository.

## Directory Structure
```
Dataset/
├── mnist/                    # MNIST dataset files
│   ├── TinyMNIST.zip        # Compressed MNIST dataset
│   └── MNIST_Dataset Description.pdf  # Dataset description
├── cancer/                   # Cancer diagnosis dataset
│   ├── Tiny Cancer Diagnosis dataset.csv  # Features
│   └── Cancer Diagnosis label.csv         # Labels
├── iris/                     # Iris dataset
│   └── Iris.csv             # Iris dataset with features and labels
└── emotion/                  # Emotion detection dataset
    └── [emotion dataset files]
```

## Usage
Each dataset is organized in its own directory and can be used with the corresponding project:

- MNIST dataset: Used in MLP & RBF project
- Cancer dataset: Used in Classification projects
- Iris dataset: Used in Clustering and Classification projects
- Emotion dataset: Used in Facial Expression Emotion Recognition project

## Data Format
- CSV files contain comma-separated values
- ZIP files contain compressed datasets
- PDF files contain dataset descriptions and documentation 
