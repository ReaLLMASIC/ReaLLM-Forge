# Core Array 

## Overview

This project implements and tests a parameterized `core_array` module using SystemVerilog. The project includes the following key components:
- **core_array.sv**: The SystemVerilog implementation of the core array module.
- **core_tb.sv**: The SystemVerilog testbench used to verify the functionality of the core array.
- **core_array.lst**: A list file used for compiling or synthesizing the SystemVerilog design, including necessary dependencies and file references.

## Modules

### core_array.lst
This file provides a configuration for compiling or synthesizing the `core_array` module. 

### core_array.sv
The `core_array.sv` file defines a configurable module for managing data flow in a 2D array of processing cores. Key components include:
- **Data Flow Control:** Manages horizontal (`hlink`) and vertical (`vlink`) data flow, with special handling ensure correct data routing at edges and corners.
- **Global Bus Management:** Coordinates data reads, writes, and addresses across the array. Initializing before processing and managing valid read/write signals to synchronize communication across cores.
- **Core Instantiation:** Generates `core_top` modules for each core, connecting them to system-wide signals like clock, reset, and bus/link signals.

### core_tb.sv
The `core_tb` testbench is designed to simulate the `core_array` module. Key features include:
- **Global Bus and Core-to-Core Link**: Simulates the global communication between cores.
- **Core Memory and Cache**: Tests the memory and caching mechanisms within the core array.
- **Local Buffers**: Verifies the behavior of local buffers and ensures data integrity.

