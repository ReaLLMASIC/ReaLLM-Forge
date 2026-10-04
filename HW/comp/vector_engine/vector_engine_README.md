# Vector Engines

This folder contains a collection of vector engines for neural network operations. These modules include floating-point adders, normalization units, and other essential components for efficient neural network processing.

## Modules Overview

### Layer_norm
#### `adder_tree.v`
- Implements an integer adder and a tree-based floating-point adder using pipelined stages.

#### `RMS_layer_norm.v`
- Performs root-mean-square (RMS) layer normalization using floating-point arithmetic.
- **Formula:**
  
  $`RMS Layer Norm(x_i) = \frac{x_i}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}}`$
  
  Where $`x_i`$ is the input vector, $`N`$ is the number of elements in the vector, and $`\epsilon`$ is a small constant for numerical stability.


#### `layer_norm.v`
- Implements standard layer normalization with parameter handling and lookup table integration.
- **Formula:**
  
  $`Layer Norm(x_i) = \frac{x_i - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta`$
  
  Where $`x_i`$ is the input vector, $`\epsilon`$ is a small constant for numerical stability, $`\gamma`$ and $`\beta`$ are learnable affine transform parameters of normalized shape.


#### `buffer_RMS.v`
- Implements a parameterized FIFO buffer with multiple input and output widths.

### Elementwise_add
#### `elem_fp_add.v` and `define.vh`
- Performs element-wise floating-point addition of two input data sets using a pipelined architecture.
- **Formula:**
  
  $`Elementwise Add(a_i, b_i) = a_i + b_i \quad \text{for} \quad i = 1, 2, \ldots, N`$
  
  Where $`a_i`$ and $`b_i`$ are elements from two input arrays of size $`N`$. The operation produces a new array where each element is the sum of corresponding elements from the input arrays.


### Softmax
#### `softmax.v` and `softermax.v`
- Implements the softmax operation, which normalizes a set of input values to a probability distribution.
- **Formula:**
  
  $`Softmax(S_i) = \frac{e^{S_i - \beta}}{\sum_{i}e^{S_i - \beta}}`$
  
  Where $`x_i`$ is the input element and $`N`$ is the total number of elements. The `softermax` may include variations for improved numerical stability or other optimizations.


### Consmax
#### `consmax.v` and `consmax_copy.v`
- Implements the consmax operation, computes the maximum value in a set of input values using an efficient comparison network or tree structure.
- **Formula:**
  
  $`ConSmax(S_i) = \frac{e^{S_i - \beta}}{\gamma}`$
  
  
  This module calculates the maximum value from a set of inputs $`( x_1, x_2, \ldots, x_n )`$ by comparing each value to find the highest one.


