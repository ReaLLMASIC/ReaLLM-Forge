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

module layer_norm #(
    parameter sig_width = 10,
    parameter exp_width = 5,
    parameter ieee_compliance = 0,
    parameter en_ubr_flag = 0,
    parameter INV_LN_NUM = 1023410176, // value of 1/LN_NUM in fp8
    parameter DATA_DEPTH = `ARR_GBUS_DATA/`ARR_IDATA_BIT,
    parameter FP_WIDTH = `LN_FP_W,
    parameter INT_WIDTH = `ARR_IDATA_BIT,
    parameter LN_NUM = `N_MODEL,
    parameter LUT_DEPTH = `SEQ_LENGTH
)( 
    input clk, rst_n,
    input  [DATA_DEPTH-1:0][INT_WIDTH-1:0] input_data ,
    //input  [FP_WIDTH-1:0] gamma ,
    //input  [FP_WIDTH-1:0] beta ,
    input  valid_in,

    //input signal of KV Cache
    input lut_wen,
    input [$clog2(LUT_DEPTH)-1:0] lut_addr,
    input [2*FP_WIDTH-1:0] lut_wdata,
    input lut_ren,
    output [2*FP_WIDTH-1:0] lut_rdata,

    output wire valid_out,
    output reg [DATA_DEPTH-1:0][INT_WIDTH-1:0] ln_out 
); 
   
endmodule
