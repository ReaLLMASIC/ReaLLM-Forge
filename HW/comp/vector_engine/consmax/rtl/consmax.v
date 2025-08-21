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

// =============================================================================
// ConSmax Top Module

module consmax#(
    parameter   IDATA_BIT = 8,  // Input Data in INT
    parameter   ODATA_BIT = 8,  // Output Data in INT
    parameter   CDATA_BIT = 8,  // Global Config Data

    parameter   EXP_BIT = 8,    // Exponent
    parameter   MAT_BIT = 7,    // Mantissa
    parameter   LUT_DATA  = EXP_BIT + MAT_BIT + 1,  // LUT Data Width (in FP)
    parameter   LUT_ADDR  = IDATA_BIT >> 1,         // LUT Address Width
    parameter   LUT_DEPTH = 2 ** LUT_ADDR           // LUT Depth for INT2FP
)(
    // Global Signals
    input                       clk,
    input                       rstn,

    // Control Signals
    input       [CDATA_BIT-1:0] cfg_consmax_shift,

    // LUT Interface
    input       [LUT_ADDR:0]    lut_waddr,
    input                       lut_wen,
    input       [LUT_DATA-1:0]  lut_wdata,

    // Data Signals
    input       [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0][`ARR_IDATA_BIT-1:0] idata,
    input       [`ARR_HNUM-1:0]                                                        idata_valid,
    output  reg [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0][`ARR_IDATA_BIT-1:0] odata,
    output      [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0]                     odata_valid
);
    wire    [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0][`ARR_IDATA_BIT-1:0] odata_w;

    genvar i,j;
    generate
        for(i=0;i<`ARR_HNUM;i++) begin
            for(j=0;j<`ARR_GBUS_DATA/`ARR_IDATA_BIT;j++) begin
                consmax_block #(
                    .IDATA_BIT(`ARR_IDATA_BIT),
                    .ODATA_BIT(`ARR_IDATA_BIT),
                    .CDATA_BIT(`ARR_CDATA_BIT),
                    .EXP_BIT(8),
                    .MAT_BIT(7)
                ) consmax_instance(
                    .clk(clk),
                    .rstn(rstn),
                    .cfg_consmax_shift(cfg_consmax_shift),
                    .lut_waddr(lut_waddr),//spi1
                    .lut_wen(lut_wen),//spi1
                    .lut_wdata(lut_wdata),//spi1
                    .idata(idata[i][j]),
                    .idata_valid(idata_valid[i]),
                    .odata(odata[i][j]),
                    .odata_valid(odata_valid[i][j])
                );
            end
        end
    endgenerate

endmodule

module consmax_block #(
    parameter   IDATA_BIT = 8,  // Input Data in INT
    parameter   ODATA_BIT = 8,  // Output Data in INT
    parameter   CDATA_BIT = 8,  // Global Config Data

    parameter   EXP_BIT = 8,    // Exponent
    parameter   MAT_BIT = 7,    // Mantissa
    parameter   LUT_DATA  = EXP_BIT + MAT_BIT + 1,  // LUT Data Width (in FP)
    parameter   LUT_ADDR  = IDATA_BIT >> 1,         // LUT Address Width
    parameter   LUT_DEPTH = 2 ** LUT_ADDR           // LUT Depth for INT2FP
)(
    // Global Signals
    input                       clk,
    input                       rstn,

    // Control Signals
    input       [CDATA_BIT-1:0] cfg_consmax_shift,

    // LUT Interface
    input       [LUT_ADDR:0]    lut_waddr,          // bitwidth + 1 for two LUTs
    input                       lut_wen,
    input       [LUT_DATA-1:0]  lut_wdata,

    // Data Signals
    input       [IDATA_BIT-1:0] idata,
    input                       idata_valid,
    output  reg [ODATA_BIT-1:0] odata,
    output  reg                 odata_valid
);

    reg       [LUT_ADDR:0]    lut_waddr_reg;
    reg                       lut_wen_reg;
    reg       [LUT_DATA-1:0]  lut_wdata_reg;
    
    always @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            lut_waddr_reg<='0;
            lut_wen_reg <= 1'b0;
            lut_wdata_reg <= '0;
        end
        else begin
            lut_waddr_reg<=lut_waddr;
            lut_wen_reg <= lut_wen;
            lut_wdata_reg <= lut_wdata;
        end
    end

    // Clock Gating for Input
    reg     [IDATA_BIT-1:0] idata_reg;
    reg                     idata_valid_reg;
    reg     [CDATA_BIT-1:0] cfg_consmax_shift_reg;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            idata_reg <= 'd0;
        end
        else if (idata_valid) begin
            idata_reg <= idata;
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            cfg_consmax_shift_reg <= 'd0;
        end
        else begin
            cfg_consmax_shift_reg <= cfg_consmax_shift;
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            idata_valid_reg <= 1'b0;
        end
        else begin
            idata_valid_reg <= idata_valid;
        end
    end

    // LUT Initialization: Convert INT to FP
    wire    [LUT_ADDR-1:0]  lut_addr    [0:1];
    wire    [LUT_DATA-1:0]  lut_rdata   [0:1];
    wire                    lut_ren;
    reg                     lut_rvalid;
    wire                    lut_wen_array [0:1];
    assign  lut_addr[0] = lut_wen_reg && ~lut_waddr_reg[LUT_ADDR] ? lut_waddr_reg[LUT_ADDR-1:0] :
                                                            idata_reg[LUT_ADDR-1:0];
    assign  lut_addr[1] = lut_wen_reg &&  lut_waddr_reg[LUT_ADDR] ? lut_waddr_reg[LUT_ADDR-1:0] :
                                                            idata_reg[LUT_ADDR+:LUT_ADDR];
    assign  lut_ren     = lut_wen_reg ? 1'b0 : idata_valid_reg;
    assign lut_wen_array[0] = lut_wen_reg && ~lut_waddr_reg[LUT_ADDR];
    assign lut_wen_array[1] = lut_wen_reg &&  lut_waddr_reg[LUT_ADDR];
    genvar i;
    generate
        for (i = 0; i < 2; i = i + 1) begin: gen_conmax_lut
            consmax_lut lut_inst (
                .ickwp0(clk),
                .iwenp0(lut_wen_array[i]),
                .iawp0(lut_addr[i]),
                .idinp0(lut_wdata_reg),
                .ickrp0(clk),
                .irenp0(lut_ren),
                .iarp0(lut_addr[i]),
                .iclkbyp(1'b1),
                .imce(1'b0),
                .irmce(2'b0),
                .ifuse(1'b1),
                .iwmce(4'b0),
                .odoutp0(lut_rdata[i])
            );
        end
    endgenerate

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            lut_rvalid <= 1'b0;
        end
        else begin
            lut_rvalid <= lut_ren;
        end
    end

    // FP Multiplication: Produce exp(Si)
    wire    [LUT_DATA-1:0]  lut_product;

    DW_fp_mult #(MAT_BIT, EXP_BIT, 0, 0)
                mult_fp ( .a(lut_rdata[0]), .b(lut_rdata[1]), .rnd(3'b000), .z(lut_product));

    // Convert FP to INT
    wire    [ODATA_BIT-1:0] odata_comb;
    reg     odata_valid_comb;

    fp2int  #(.EXP_BIT(EXP_BIT), .MAT_BIT(MAT_BIT), .ODATA_BIT(ODATA_BIT), .CDATA_BIT(CDATA_BIT)) fp2int_inst (
        .clk                    (clk),
        .rstn                   (rstn),
        .cfg_consmax_shift      (cfg_consmax_shift_reg),
        .idata                  (lut_product),
        .odata                  (odata_comb)
    );

    // Output
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            odata <= 'd0;
        end
        else if (lut_rvalid) begin
            odata <= odata_comb;
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            odata_valid_comb <= 1'b0;
            odata_valid <= 1'b0;
        end
        else begin
            odata_valid_comb <= lut_rvalid;
            odata_valid <= odata_valid_comb;
        end
    end

endmodule

// =============================================================================
// Floating-Point to Integer Converter

module fp2int #(
    parameter   EXP_BIT = 8,
    parameter   MAT_BIT = 7,

    parameter   IDATA_BIT = EXP_BIT + MAT_BIT + 1,  // FP-Input
    parameter   ODATA_BIT = 8,  // INT-Output
    parameter   CDATA_BIT = 8   // Config
)(
    // Control Signals
    input   [CDATA_BIT-1:0] cfg_consmax_shift,
    input clk,
    input rstn,
    // Data Signals
    input   [IDATA_BIT-1:0] idata,
    output  [ODATA_BIT-1:0] odata
);

    localparam  EXP_BASE = 2 ** (EXP_BIT - 1) - 1;

    // Extract Sign, Exponent and Mantissa Field
    reg                     idata_sig,idata_sig_reg;
    reg     [EXP_BIT-1:0]   idata_exp;
    reg     [MAT_BIT:0]     idata_mat;

    always @(*) begin
        idata_sig = idata[IDATA_BIT-1];
        idata_exp = idata[MAT_BIT+:EXP_BIT];
        idata_mat = {1'b1, idata[MAT_BIT-1:0]};
    end

    // Shift and Round Mantissa to Integer
    reg     [MAT_BIT:0]     mat_shift,mat_shift_reg;
    reg     [MAT_BIT:0]     mat_round,mat_round_reg;

    always @(*) begin
        if (idata_exp >= EXP_BASE) begin    // >= 1.0
            if (MAT_BIT <= (cfg_consmax_shift + (idata_exp - EXP_BASE))) begin // Overflow
                mat_shift = {(MAT_BIT){1'b1}};
                mat_round = mat_shift;
            end
            else begin
                mat_shift = idata_mat >> (MAT_BIT - cfg_consmax_shift - (idata_exp - EXP_BASE));
                mat_round = mat_shift[MAT_BIT:1] + mat_shift[0];
            end
        end
        else begin  // <= 1.0
            if (cfg_consmax_shift < (EXP_BASE - idata_exp)) begin // Underflow
                mat_shift = {(MAT_BIT){1'b0}};
                mat_round = mat_shift;
            end
            else begin
                mat_shift = idata_mat >> (MAT_BIT - cfg_consmax_shift + (EXP_BASE - idata_exp));
                mat_round = mat_shift[MAT_BIT:1] + mat_shift[0];
            end
        end
    end
    always @(posedge clk or negedge rstn) begin
        if(~rstn) begin
            idata_sig_reg<='0;
            mat_shift_reg<='0;
            mat_round_reg<='0;
        end
        else begin
            idata_sig_reg<=idata_sig;
            mat_shift_reg<=mat_shift;
            mat_round_reg<=mat_round;
        end
    end
    
    // Convert to 2's Complementary Integer
    assign  odata = {idata_sig_reg, idata_sig_reg ? (~mat_round_reg[MAT_BIT-:ODATA_BIT] + 1'b1) : 
                                              mat_round_reg[MAT_BIT-:ODATA_BIT]};

endmodule
