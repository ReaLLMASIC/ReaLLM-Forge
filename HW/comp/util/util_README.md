# Utility Modules 

This folder contains various Verilog and SystemVerilog modules designed to facilitate different aspects of nanoGPT design. These modules cover functionalities like floating-point arithmetic, memory operations, FIFO buffers, counters, and more. Below is a detailed description of each module:

## Module Descriptions

### align.v
This module handles the alignment of input data. It includes logic for adjusting the position of bits or bytes within a data word, which is useful in applications like data serialization or parallel processing.

### define.vh
This header file contains macro definitions used across multiple modules. 

### fadd.sv
The fadd module implements a floating-point addition unit. It adheres to the IEEE 754 standard for floating-point arithmetic, supporting addition operations on single-precision (32-bit) or double-precision (64-bit) floating-point numbers. The module handles normalization, rounding, and special cases like NaN and infinity.

### fifo.v
This module implements a basic First-In-First-Out (FIFO) buffer, ensuring data is read in the order it was written. 

### mem.v
The mem.v module defines a memory block, either RAM or ROM. It supports data storage and allows for read and write operations. This module can be parameterized for different memory sizes and configurations, offering flexibility for various applications.

### pe.v
The Processing Element (PE) module is a core unit used in parallel computing architectures. It performs computations or data processing tasks and can be part of larger systems like SIMD (Single Instruction, Multiple Data) or MIMD (Multiple Instruction, Multiple Data) architectures.

### rr_arbiter.sv
The Round-Robin Arbiter (rr_arbiter.sv) module implements a round-robin scheduling algorithm. It manages access to a shared resource among multiple requesters in a fair manner, ensuring that each requester gets equal opportunity over time.

### sync_fifo_bak.sv
This module represents a synchronized FIFO with a backup mechanism. It handles data buffering between different clock domains or when synchronization is required. The backup mechanism ensures data integrity and reliability.

### sync_fifo_data.sv
Similar to sync_fifo_bak.sv, this module focuses on synchronizing data across clock domains. It ensures proper data sequencing and prevents data loss or corruption during asynchronous data transfers.

### counter.sv
The counter.sv module implements a counter, which can be used for counting events, timing, or as a simple state machine. It may include features like up/down counting, loadable values, and overflow detection, making it versatile for various applications.

### fmul.sv
The fmul module provides a floating-point multiplication unit. It adheres to the IEEE 754 standard and supports multiplication operations for single-precision or double-precision floating-point numbers. The module handles normalization, rounding, and special cases such as NaN and infinity.

