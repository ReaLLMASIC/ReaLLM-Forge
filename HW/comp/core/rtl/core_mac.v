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

// MAC Top Module

module core_mac #(
    parameter   MAC_NUM   = 64, // MAC Line Size
    parameter   IDATA_BIT = 8,  // Input
    parameter   ODATA_BIT = IDATA_BIT*2+$clog2(MAC_NUM) // Output
)(
    // Global Signals
    input                               clk,
    input                               rst,

    // Data Signals
    input       [IDATA_BIT*MAC_NUM-1:0] idataA,
    input       [IDATA_BIT*MAC_NUM-1:0] idataB,
    input                               idata_valid,
    output      [ODATA_BIT-1:0]         odata,
    output                              odata_valid
);
    // Multiplication
    wire    [IDATA_BIT*2*MAC_NUM-1:0]   product;
    wire                                product_valid;

    mul_line    #(.MAC_NUM(MAC_NUM), .IDATA_BIT(IDATA_BIT)) mul_inst (
        .clk                            (clk),
        .rst                            (rst),
        .idataA                         (idataA),
        .idataB                         (idataB),
        .idata_valid                    (idata_valid),
        .odata                          (product),
        .odata_valid                    (product_valid)
    );

    // Addition
    adder_tree  #(.MAC_NUM(MAC_NUM), .IDATA_BIT(IDATA_BIT*2)) adt_inst (
        .clk                            (clk),
        .rst                            (rst),
        .idata                          (product),
        .idata_valid                    (product_valid),
        .odata                          (odata),
        .odata_valid                    (odata_valid)
    );  

endmodule

// =============================================================================
// MUL Line

module mul_line #(
    parameter   MAC_NUM = 64,
    parameter   IDATA_BIT = 8,
    parameter   ODATA_BIT = IDATA_BIT * 2
)(
    // Global Signals
    input                               clk,
    input                               rst,
    
    // Data Signals
    input       [IDATA_BIT*MAC_NUM-1:0] idataA,
    input       [IDATA_BIT*MAC_NUM-1:0] idataB,
    input                               idata_valid,
    output  reg [ODATA_BIT*MAC_NUM-1:0] odata,
    output  reg                         odata_valid
);

    // Input Gating
    reg     [IDATA_BIT-1:0] idataA_reg  [0:MAC_NUM-1];
    reg     [IDATA_BIT-1:0] idataB_reg  [0:MAC_NUM-1];

    genvar i;
    generate
        for (i = 0; i < MAC_NUM; i = i + 1) begin: gen_mul_input
            always @(posedge clk or posedge rst) begin
                if (rst) begin
                    idataA_reg[i] <= 'd0;
                    idataB_reg[i] <= 'd0;
                end
                else if (idata_valid) begin
                    idataA_reg[i] <= idataA[i*IDATA_BIT+:IDATA_BIT];
                    idataB_reg[i] <= idataB[i*IDATA_BIT+:IDATA_BIT];
                end
            end
        end
    endgenerate

    // Mutiplication
    wire    [ODATA_BIT-1:0] product [0:MAC_NUM-1];

    generate
        for (i = 0; i < MAC_NUM; i = i + 1) begin: gen_mul
            mul_int #(.IDATA_BIT(IDATA_BIT), .ODATA_BIT(ODATA_BIT)) mul_inst (
                .idataA                 (idataA_reg[i]),
                .idataB                 (idataB_reg[i]),
                .odata                  (product[i])
            );
        end
    endgenerate

    // Output
    generate
        for (i = 0; i < MAC_NUM; i = i + 1) begin: gen_mul_output
            always @(*) begin
                odata[i*ODATA_BIT+:ODATA_BIT] = product[i];
            end
        end
    endgenerate

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata_valid <= 'd0;
        end
        else begin
            odata_valid <= idata_valid;
        end
    end

endmodule

// =============================================================================
// Configurable Adder Tree. Please double-check it's synthesizable.

module adder_tree #(
    parameter   MAC_NUM = 64,
    parameter   IDATA_BIT = 16,
    parameter   ODATA_BIT = IDATA_BIT + $clog2(ODATA_BIT)
)(
    // Global Signals
    input                               clk,
    input                               rst,

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
                always @(posedge clk or posedge rst) begin
                    if (rst) begin
                        add_valid <= 1'b0;
                    end
                    else begin
                        add_valid <= idata_valid;
                    end
                end
            end
            else if (i % 2 == 1'b0) begin   // Even Stage, Insert a pipeline, Start from 0, 2, 4...
                always @(posedge clk or posedge rst) begin
                    if (rst) begin
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
            localparam  OUT_BIT = IDATA_BIT + (i + 1'b1);
            localparam  OUT_NUM = MAC_NUM  >> (i + 1'b1);

            reg     [OUT_BIT-2:0]   add_idata   [0:OUT_NUM*2-1];
            wire    [OUT_BIT-1:0]   add_odata   [0:OUT_NUM-1];

            for (j = 0; j < OUT_NUM; j = j + 1) begin: gen_adt_adder
                
                // Organize adder inputs
                if (i == 0) begin   // Input Stage
                    always @(posedge clk or posedge rst) begin
                        if (rst) begin
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
                    always @(posedge clk or posedge rst) begin
                        if (rst) begin
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
                add_int #(.IDATA_BIT(OUT_BIT-1), .ODATA_BIT(OUT_BIT)) adder_inst (
                    .idataA                 (add_idata[j*2]),
                    .idataB                 (add_idata[j*2+1]),
                    .odata                  (add_odata[j])
                );
            end
        end
    endgenerate

    // Output
    always @(*) begin
        odata       = gen_adt_stage[STAGE_NUM-1].add_odata[0];
        odata_valid = gen_adt_valid[STAGE_NUM-1].add_valid;
    end

endmodule