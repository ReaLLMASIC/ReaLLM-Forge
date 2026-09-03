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

module add_int #(
    parameter   IDATA_BIT = 8,
    parameter   ODATA_BIT = 9,
    parameter   sig_width = 4,
    parameter   exp_width = 3,
    parameter   ieee_compliance = 0
)(
    // Data Signals
    input       [IDATA_BIT-1:0] idataA,
    input       [IDATA_BIT-1:0] idataB,
    output      [ODATA_BIT-1:0] odata
);

    reg signed  [ODATA_BIT-1:0] odata_comb;

    always @(*) begin
        odata_comb = $signed(idataA) + $signed(idataB);
    end

    assign  odata = odata_comb;

endmodule

module adder_tree1 #(
    parameter   MAC_NUM = 8,
    parameter   IDATA_BIT = 8,
    parameter   ODATA_BIT = IDATA_BIT, //MAC_NUM? was $clog2(ODATA_BIT) before
    parameter   sig_width = 23,
    parameter   exp_width = 8,
    parameter   ieee_compliance = 0
)(
    // Global Signals
    input                               clk,
    input                               rstn,

    // Data Signals
    input       [IDATA_BIT*MAC_NUM-1:0] idata,
    input                               idata_valid,
    output  reg [ODATA_BIT-1:0]         odata,
    output  reg                         odata_valid
);

    localparam  STAGE_NUM = $clog2(MAC_NUM);

    // Insert a pipeline every two stages
    // Validation
    genvar i, j;
    generate
        for (i = 0; i < STAGE_NUM; i = i + 1) begin: gen_adt_valid
            reg             add_valid;

            if (i == 0) begin   // Input Stage
                always @(posedge clk or negedge rstn) begin
                    if (!rstn) begin
                        add_valid <= 1'b0;
                    end
                    else begin
                        add_valid <= idata_valid;
                    end
                end
            end
            else if (i % 2 == 1'b0) begin   // Even Stage, Insert a pipeline, Start from 0, 2, 4...
                always @(posedge clk or negedge rstn) begin
                    if (!rstn) begin
                        add_valid <= 1'b0;
                    end
                    else begin
                        add_valid <= gen_adt_valid[i-1].add_valid;
                    end
                end
            end
            else begin  // Odd Stage, Combinational, Start from 1, 3, 5...
                always @(*) begin
                    add_valid = gen_adt_valid[i-1].add_valid;
                end
            end
        end
    endgenerate

    // Adder
    generate
        for (i = 0; i <STAGE_NUM; i = i + 1) begin: gen_adt_stage
            localparam  OUT_BIT = IDATA_BIT;
            localparam  OUT_NUM = MAC_NUM  >> (i + 1'b1);

            reg     [OUT_BIT-1:0]   add_idata   [0:OUT_NUM*2-1];
            wire    [OUT_BIT-1:0]   add_odata   [0:OUT_NUM-1];

            for (j = 0; j < OUT_NUM; j = j + 1) begin: gen_adt_adder

                // Organize adder inputs
                if (i == 0) begin   // Input Stage
                    always @(posedge clk or negedge rstn) begin
                        if (!rstn) begin
                            add_idata[j*2]   <= 'd0;
                            add_idata[j*2+1] <= 'd0;
                        end
                        else if (idata_valid) begin
                            add_idata[j*2]   <= idata[(j*2+0)*IDATA_BIT+:IDATA_BIT];
                            add_idata[j*2+1] <= idata[(j*2+1)*IDATA_BIT+:IDATA_BIT];
                        end
                    end
                end
                else if (i % 2 == 0) begin  // Even Stage, Insert a pipeline
                    always @(posedge clk or negedge rstn) begin
                        if (!rstn) begin
                            add_idata[j*2]   <= 'd0;
                            add_idata[j*2+1] <= 'd0;
                        end
                        else if (gen_adt_valid[i-1].add_valid) begin
                            add_idata[j*2]   <= gen_adt_stage[i-1].add_odata[j*2];
                            add_idata[j*2+1] <= gen_adt_stage[i-1].add_odata[j*2+1];
                        end
                    end
                end
                else begin  // Odd Stage, Combinational
                    always @(*) begin
                        add_idata[j*2]   = gen_adt_stage[i-1].add_odata[j*2];
                        add_idata[j*2+1] = gen_adt_stage[i-1].add_odata[j*2+1];
                    end
                end

                // Adder instantization
                //assign add_odata[j] = add_idata[j*2] + add_idata[j*2+1];
                reg [7:0] status_add_1;
                DW_fp_add #(sig_width, exp_width, ieee_compliance)
                    add1 ( .a(add_idata[j*2]), .b(add_idata[j*2+1]), .rnd(3'b000), .z(add_odata[j]), .status(status_add_1) );
                
            end
        end
    endgenerate

    // Output
    always @(*) begin
        odata       = gen_adt_stage[STAGE_NUM-1].add_odata[0];
        odata_valid = gen_adt_valid[STAGE_NUM-1].add_valid;
    end
endmodule 
