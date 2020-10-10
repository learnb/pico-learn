# Pico-Learn

A Pico-8 library for graph-based learning.

## Features

### Neural Networks

Provides functions to create and train fully-connected neural networks. The interface for creating layers and structuring networks is inspired by various python machine learning frameworks.

Two object classes are provided: "nn" and "layer"

### Num-Pico

A numerical library inspired by NumPy providing functions for common vector and matrix operations. Used heavily by the NN lib.

Num-Pico provides the following functions:

*Vectors*

| Method      | Description                                          |
| ----------- | ---------------------------------------------------- |
| np_argmax   | returns index of max value in vector                 |
| np_vec_add  | component-wise vector addition                       |
| np_vec_sub  | component-wise vector subtraction                    |
| np_vec_mutl | component-wise vector multiplication                 |
| np_vec_rand | return n-dimension vector with random values (-5, 5) |
| np_vec_func | applies function to each vector component            |

*Matrices*

| Method      | Description |
| ----------- | ----------- |
| np_mat_rand   | returns n-dimension matrix with random values (-5, 5) |
| np_dot  | dot product of 2 matrices, A & B. A is m-by-n, B is n-by-p |
| np_mv_dot  | dot product of vector and n-by-p matrix |
| np_vm_dot | dot product of n-by-p matrix and vector |


## Todo

- Create set of demos of controlling in-game character using pico-learn lib
  - AI agent following target position on screen
  - AI agent playing tic-tac-toe
- QLearning 
  - Add functions to support deep q-learning experiments
