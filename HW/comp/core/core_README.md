# Core Components

This folder contains the core hardware components for the nanoGPT project. The modules in this directory are responsible for various fundamental operations such as accumulation, buffering, memory management, and quantization, which are crucial for the hardware-accelerated implementation of the nanoGPT model.

## Overview

**core_acc.v**  
   Implements the accumulation functionality for the hardware. This module is responsible for accumulating partial results during the processing of data within the GPT model.

**core_buf.v**  
   This module acts as a buffer for temporary data storage during processing. It helps manage data flow between various components of the hardware system.

**core_mac.v**  
   The multiply-accumulate (MAC) module is essential for performing efficient matrix multiplications, which are a core operation in neural networks like GPT.

**core_mem.v**  
   Manages memory operations, providing read/write access to the on-chip or external memory. This module is critical for feeding data into the computation pipeline.

**core_quant.v**  
   Handles quantization operations, converting floating-point data into reduced-precision formats to speed up processing and reduce hardware resource consumption.

**core_top.v**  
   The top-level module that integrates all core components. It coordinates the operation of the entire hardware core, ensuring that the various components work together efficiently.

**core_top.lst**
   List of signals and I/O for the top-level module
