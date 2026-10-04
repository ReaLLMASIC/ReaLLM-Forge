// Copyright (c) 2024, Saligane's Group at University of Michigan and Google Research
//
// Licensed under the Apache License, Version 2.0 (the "License");

// you may not use this file except in compliance with the License.

// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module core_quant #(
    parameter   IDATA_BIT = 32,
    parameter   ODATA_BIT = 8
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Global Config Signals
    input       [IDATA_BIT-1:0] cfg_quant_scale,
    input       [IDATA_BIT-1:0] cfg_quant_bias,
    input       [IDATA_BIT-1:0] cfg_quant_shift,

    // Data Signals
    input       [IDATA_BIT-1:0] idata,
    input                       idata_valid,
    output  reg [ODATA_BIT-1:0] odata,
    output  reg                 odata_valid
);

    // Input Gating
    // Causing the input from the accumulator register, no pipeline needed here

    // Quantize: Scale x Input + Bias
    reg signed  [IDATA_BIT*2-1:0]   quantized_product;
    reg                             quantized_product_valid;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            quantized_product <= 'd0;
        end
        else if (idata_valid) begin
            quantized_product <= $signed(idata) * $signed(cfg_quant_scale) + $signed(cfg_quant_bias);
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            quantized_product_valid <= 1'b0;
        end
        else begin
            quantized_product_valid <= idata_valid;
        end
    end

    // Quantize: Shift and Round
    reg signed  [IDATA_BIT*2-1:0]   quantized_shift;
    reg signed  [IDATA_BIT*2-1:0]   quantized_round;

    always @(*) begin
        quantized_shift = quantized_product >> cfg_quant_shift;
    end

    always @(*) begin
        quantized_round = $signed(quantized_shift[IDATA_BIT*2-1:1]) + 
                          $signed({quantized_shift[IDATA_BIT*2-1], quantized_shift[0]});
    end

    // Quantize: Detect Overflow
    reg         [ODATA_BIT-1:0]     quantized_overflow;

    always @(*) begin
        if ((quantized_round[IDATA_BIT*2-1] ^ (&quantized_round[IDATA_BIT*2-2:ODATA_BIT-1])) ||
            (quantized_round[IDATA_BIT*2-1] ^ (|quantized_round[IDATA_BIT*2-2:ODATA_BIT-1]))) begin
            quantized_overflow = {quantized_round[IDATA_BIT*2-1], 
                  {(ODATA_BIT-1){~quantized_round[IDATA_BIT*2-1]}}};
        end
        else begin
            quantized_overflow = {quantized_round[IDATA_BIT*2-1], quantized_round[ODATA_BIT-2:0]};
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata <= 'd0;
        end
        else if (quantized_product_valid) begin
            odata <= quantized_overflow;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata_valid <= 'd0;
        end
        else begin
            odata_valid <= quantized_product_valid;
        end
    end

endmodule