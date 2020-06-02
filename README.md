# Pico-Learn

A Pico-8 framework for graph-based learning.

## Features

### Neural Networks

Provides functions to create and train fully-connected neural networks. The interface for creating layers and structuring networks is inspired by various python machine learning frameworks.

Two object classes are provided: "nn" and "layer"

### NumPico

A numerical library inspired by NumPy providing functions for common vector and matrix operations. Used heavily by the NN lib.

### Visuals

Display NN structure and heatmap for parameter values.

Display training stats.



## Todo

- UI for Dashboard / Classification Demo
  - enable mouse cursor
  - Make visuals "windowed"
  - Selectable datasets
  - Changeable NN structure
  - Changeable hyperparameters (learning rate, activation)
  - make graph for training loss
- NN lib
  - Test for correctness on classification problems
- QLearning 
  - Add functions to support deep q-learning experiments
