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

`ifndef __ELEM_FP_ADD_V_
`define __ELEM_FP_ADD_V_

// macros
// bit width
`ifndef DATA_SIZE
    `define DATA_SIZE   8
`endif
`ifndef ADDR_SIZE
    `define ADDR_SIZE   20
`endif
// input config
`ifndef INPUT_NUM
    `define INPUT_NUM   12
`endif
`ifndef ROW_NUM
    `define ROW_NUM     24
`endif
// address
`ifndef ADDR_BASE
    `define ADDR_BASE   0
`endif
`ifndef ADDR_OFFU
    `define ADDR_OFFU   1'b1
`endif

//----------------------------------------------------------------------------
//  Input ports         Size & Description
//  ===============     ==================
//  rst                 1
//                      async high reset
//
//  in_data_valid       1
//                      indicating valid input
//
//  data_in             (INPUT_NUM * DATA_SIZE)
//                      INPUT_NUM elements input each cycle
//
//  org_data_in         (ROW_NUM * DATA_SIZE)
//                      ROW_NUM elements input each cycle
//
//  Output ports        Size & Description
//  ===============     ==================
//  res_out             (INPUT_NUM * DATA_SIZE)
//                      INPUT_NUM elements output each cycle
//
//  org_data_addr       (ADDR_SIZE)
//                      SRAM adderess output
//
//  org_data_rd         1
//                      SRAM read en
//
//----------------------------------------------------------------------------
module elem_add(
    input clk, rst,

    // input addr_reset,
    input in_data_valid,
    input [`INPUT_NUM-1:0][`DATA_SIZE-1:0] data_in,
    input [`ROW_NUM-1:0][`DATA_SIZE-1:0] org_data_in,

    output reg [`INPUT_NUM-1:0][`DATA_SIZE-1:0] res_out,
    output reg [`ADDR_SIZE-1:0] org_data_addr,
    output org_data_rd
);
    localparam BLK_NUM = `ROW_NUM / `INPUT_NUM;

    initial begin: param_check
        if(`ROW_NUM < `INPUT_NUM) begin
            $display(   "[INFO][ERR]: %m\n\tROW_NUM(%0d) has to be no smaller than INPUT_NUM(%0d)", 
                        `ROW_NUM, `INPUT_NUM);
        end
        if(BLK_NUM*`INPUT_NUM != `ROW_NUM) begin
            $display(   "[INFO][ERR]: %m\n\tROW_NUM(%0d) has to be a multiple of INPUT_NUM(%0d)", 
                        `ROW_NUM, `INPUT_NUM);
        end
    end

    reg valid;
    reg [`INPUT_NUM-1:0][`DATA_SIZE-1:0] data_blk; // delay data_in
    reg [$clog2(`ROW_NUM):0] blk_cnt;

    wire new_line;
    wire [`INPUT_NUM-1:0][`DATA_SIZE-1:0] org_data_blk; // assign org_data_in
    wire [`INPUT_NUM-1:0][`DATA_SIZE-1:0] add_out;

    assign new_line = (blk_cnt==$unsigned(`ROW_NUM-`INPUT_NUM));
    assign org_data_rd = in_data_valid | valid;
    // synthesis able?
    assign org_data_blk = org_data_in[blk_cnt +: `INPUT_NUM];

    always @(posedge clk, posedge rst) begin
        if(rst) begin
            valid <= 0;
            data_blk <= 0;
            blk_cnt <= 0;
            res_out <= 0;
            org_data_addr <= `ADDR_BASE;
        end else begin
            valid <= in_data_valid;
            data_blk <= data_in;
            if(in_data_valid & ~valid) begin
                blk_cnt <= 0;
            end
            if(valid) begin
                blk_cnt <= (new_line)? 0: blk_cnt + `INPUT_NUM;
                res_out <= add_out;
                if(new_line) begin
                    org_data_addr <= org_data_addr + `ADDR_OFFU;
                end
            end
        end
    end

    DW_fp_addsub #(
        .sig_width(`MNT_SIZE),
        .exp_width(`EXP_SIZE)
    ) fadd0 [`INPUT_NUM-1:0](
        .a(data_blk),
        .b(org_data_blk),
        .rnd(3'b000),
        .op(1'b0),
        .z(add_out),
        .status()
    );

endmodule

`endif

