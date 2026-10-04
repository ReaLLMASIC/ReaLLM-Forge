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

module buffer #(
    parameter DATA_WIDTH = 8,
    parameter FIFO_DEPTH = 64,
    parameter INPUT_WIDTH = 8,  // New parameter for input width
    parameter OUTPUT_WIDTH = 8  // New parameter for output width
)(
    input clk,                                  // Clock input
    input rst_n,                                // Reset input, active high
    input wr_en,                                // Write enable
    input rd_en,                                // Read enable
    input [INPUT_WIDTH*DATA_WIDTH-1:0] data_in, // Modified for multiple inputs
    output reg [OUTPUT_WIDTH*DATA_WIDTH-1:0] data_out, // Modified for multiple outputs
    output full,                                // FIFO full flag
    output empty                                // FIFO empty flag
);

reg [DATA_WIDTH-1:0] fifo_mem [FIFO_DEPTH-1:0];
reg [$clog2(FIFO_DEPTH)-1:0] wr_ptr, rd_ptr;
reg [$clog2(FIFO_DEPTH+1)-1:0] fifo_count; // Fix: Use FIFO_DEPTH for size calculation

// Adjusted FIFO full and empty logic to consider INPUT_WIDTH and OUTPUT_WIDTH
assign full = (fifo_count > (FIFO_DEPTH - INPUT_WIDTH));
assign empty = (fifo_count < OUTPUT_WIDTH);

integer i;

// Write operation adjusted for INPUT_WIDTH
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_ptr <= 0;
    end else if (wr_en && !full) begin
        for (i = 0; i < INPUT_WIDTH; i = i + 1) begin
            if ((wr_ptr + i) < FIFO_DEPTH) begin // Ensure we don't exceed FIFO bounds
                fifo_mem[wr_ptr + i] <= data_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            end
        end
        wr_ptr <= wr_ptr + INPUT_WIDTH;
    end
end

// Read operation adjusted for OUTPUT_WIDTH
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_ptr <= 0;
        data_out <= 0;
    end else if (rd_en && !empty) begin
        for (i = 0; i < OUTPUT_WIDTH; i = i + 1) begin
            if ((rd_ptr + i) < FIFO_DEPTH) begin // Ensure we don't exceed FIFO bounds
                data_out[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= fifo_mem[rd_ptr + i];
            end
        end
        rd_ptr <= rd_ptr + OUTPUT_WIDTH;
    end
end

// Adjust FIFO count management for INPUT_WIDTH and OUTPUT_WIDTH
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fifo_count <= 0;
    end else if (wr_en && !full && !rd_en) begin
        fifo_count <= fifo_count + INPUT_WIDTH;
    end else if (rd_en && !empty && !wr_en) begin
        fifo_count <= fifo_count - OUTPUT_WIDTH;
    end else if (wr_en && rd_en && !full && !empty) begin
        fifo_count <= fifo_count + INPUT_WIDTH - OUTPUT_WIDTH;
    end
end

endmodule
