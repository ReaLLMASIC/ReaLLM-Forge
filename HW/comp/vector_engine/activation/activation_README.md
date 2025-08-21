# Activation Functions

This folder contains Verilog modules for various floating-point operations and activation functions used in nanoGPT implementations. The modules are designed to be IEEE 754 compliant and can be integrated into larger systems for performing floating-point arithmetic and activation functions.

## Module Descriptions

### `gelu.v`
This module implements GELU (Gaussian Error Linear Unit) activation function.
- **Formula:** 
$`{GELU}(x) = 0.5x \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]`$
- **Key Features:**
  - Implements the GELU function using a combination of floating-point operations.
  - Optimized for hardware implementations.

### `relu.v`
This module implements the ReLU (Rectified Linear Unit) activation function.
- **Formula:** 
$`{ReLU}(x) = max(0, x)`$
- **Key Features:**
  - Simple and efficient implementation.
  - Hardware-optimized for speed.

### `silu.v`
This module implements SiLU (Sigmoid Linear Unit) activation function. 
- **Formula:**
$`{SiLU}(x) = x \cdot \sigma(x)$
  - where $`sigma(x)`$ is the sigmoid function defined as $`sigma(x) = \frac{1}{1 + e^{-x}}`$.
- **Key Features:**
  - Combines sigmoid and linear operations.
  - Useful in various neural network architectures.

### `fadd_sub.v`
This module performs floating-point addition and subtraction operations. It handles the mantissa alignment, exponent adjustment, normalization, and rounding according to the IEEE 754 standard.

### `fdiv.v`
This module performs floating-point division. It includes mantissa normalization, exponent adjustment, and rounding. Handles division by zero by returning infinity.

### `fdiv2.v`, `fdiv4.v`
These are variations of the floating-point division module (`fdiv.v`). `fdiv2` module performs floating-point division by 2. `fdiv4` module performs floating-point division by 4.

### `fexp.v`
This module calculates the exponential function (e^x) using floating-point arithmetic. It's commonly used in activation functions like GELU and other neural network operations.


