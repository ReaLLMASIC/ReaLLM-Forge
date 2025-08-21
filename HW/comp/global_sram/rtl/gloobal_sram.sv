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

module global_sram #(
    parameter DATA_BIT = `ARR_GBUS_DATA,
    parameter DEPTH = `GLOBAL_SRAM_DEPTH
) (
    input clk,
    input [$clog2(DEPTH+1)-1:0]global_sram_addr,
    input global_sram_wen,
    input [DATA_BIT-1:0] global_sram_wdata,
    input global_sram_ren,
    output logic [DATA_BIT-1:0] global_sram_rdata
);
mem_sp  #(.DATA_BIT(DATA_BIT), .DEPTH(DEPTH)) global_sram_inst (
    .clk                    (clk),
    .addr                   (global_sram_addr),
    .wen                    (global_sram_wen),
    .wdata                  (global_sram_wdata),
    .ren                    (global_sram_ren),
    .rdata                  (global_sram_rdata)
);
endmodule
