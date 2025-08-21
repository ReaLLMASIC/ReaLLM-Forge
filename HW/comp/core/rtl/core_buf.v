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

module core_buf #(
    // Core-to-Core Link (Access Activation Buffer)
    parameter   GBUS_DATA = 64,
    
    // Activation Buffer. Avoid direct access to large memory.
    parameter   ABUF_DATA  = 64*8,                  // Data Bitwidth
    parameter   ABUF_DEPTH = 16,                    // Depth
    parameter   ABUF_ADDR  = $clog2(ABUF_DEPTH)     // Address Bitwidth
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Channel - Core-to-Core Link
    input       [GBUS_DATA-1:0] clink_wdata,
    input                       clink_wen,
    output      [GBUS_DATA-1:0] clink_rdata,
    output                      clink_rvalid,

    // Channel - Activation Buffer for MAC Operation
    //input                     abuf_mux,           // Annotate for Double-Buffering ABUF
    input       [ABUF_ADDR-1:0] abuf_waddr,
    input       [ABUF_ADDR-1:0] abuf_raddr,
    input                       abuf_ren,
    output      [ABUF_DATA-1:0] abuf_rdata,
    output  reg                 abuf_rvalid
);

    // =============================================================================
    // Memory Instantization: Dual-Port or Double-Buffering ABUF
    // TODO: Design Exploration for DP/DB and SRAM/REGFILE/DFF ABUF

    wire    [ABUF_DATA-1:0]     abuf_wdata;
    wrie                        abuf_wen;

    mem_dp  #(.DATA_BIT(ABUF_DATA), .DEPTH(ABUF_DEPTH)) abuf_inst (
        .clk                    (clk),
        .waddr                  (abuf_waddr),
        .wen                    (abuf_wen),
        .wdata                  (abuf_wdata),
        .raddr                  (abuf_raddr),
        .ren                    (abuf_ren),
        .rdata                  (abuf_rdata)
    );

    /*
    mem_db  #(.DATA_BIT(LBUF_BIT), .DEPTH(LBUF_DEPTH))  abuf_inst (
        .clk                    (clk),
        .sw                     (abuf_mux),
        .waddr                  (abuf_waddr),
        .wen                    (abuf_wen),
        .wdata                  (abuf_wdata),
        .raddr                  (abuf_raddr),
        .ren                    (abuf_ren),
        .rdata                  (abuf_rdata)
    );
    */

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            abuf_rvalid <= 1'b0;
        end
        else begin
            abuf_rvalid <= abuf_ren;
        end
    end

    // =============================================================================
    // Core-to-Core Link Channel

    // 1. Write Channel: CLINK -> Core
    reg     [GBUS_DATA-1:0]     clink_reg;
    reg                         clink_reg_valid;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            clink_reg <= 'd0;
        end
        else if (clink_wen) begin
            clink_reg <= clink_wdata;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            clink_reg_valid <= 1'b0;
        end
        else begin
            clink_reg_valid <= clink_wen;
        end
    end

    // 2. Read Channel: Core -> CLINK
    assign  clink_rdata  = clink_reg;
    assign  clink_rvalid = clink_reg_valid;

    // =============================================================================
    // ABUF Write Channel: Series to Parallel

    align_s2p   #(.IDATA_BIT(GBUS_DATA), .ODATA_BIT(ABUF_DATA)) abuf_s2p (
        .clk                    (clk),
        .rst                    (rst),
        .idata                  (clink_reg),
        .idata_valid            (clink_reg_valid),
        .odata                  (abuf_wdata),
        .odata_valid            (abuf_wen)
    );

endmodule