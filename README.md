# MNIST Neural Network Model Benchmark Notebook

This repository contains a Jupyter notebook (`main.ipynb`) focused on comparing different neural network architectures and their performance on the MNIST handwritten digit classification task.

## Notebook Overview

The `main.ipynb` notebook provides a comprehensive analysis of various neural network models with different architectures (CNN and MLP). It includes:

- Data loading and preprocessing of the MNIST dataset
- Implementation of multiple neural network architectures
- Training and evaluation procedures
- Performance visualization and benchmarking
- Model comparison and analysis

## Key Features of the Notebook

- End-to-end implementation of neural network models
- Comparative analysis of different architectures:
  - Convolutional Neural Networks (CNN) of various sizes
  - Multi-Layer Perceptrons (MLP) of various sizes
  - Models with different depths and widths
- Comprehensive performance metrics including:
  - Accuracy and loss measurements
  - Training and inference time
  - Parameter efficiency
  - Per-class performance analysis

## How to Use the Notebook

1. Open `main.ipynb` in Jupyter or any compatible notebook environment
2. Execute the cells sequentially to reproduce the analysis
3. Experiment with different model configurations as needed

## Supporting Files

The notebook relies on several supporting files:

- `benckmark.py` containing visualization and analysis functions


## Requirements

To run the notebook, you'll need:

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- tqdm
- Jupyter Notebook/Lab

## Visualizations in the Notebook

The notebook includes several visualizations to help understand model performance:

- Test accuracy comparison across models
- Per-class accuracy heatmaps
- Model complexity vs. performance trade-off analysis
- Learning curves showing training progression
- Efficiency metrics (accuracy per parameter, accuracy per time)

## Dataset

The notebook uses the MNIST dataset, a collection of 70,000 grayscale images of handwritten digits (0-9). The dataset is automatically downloaded and processed when running the notebook.
