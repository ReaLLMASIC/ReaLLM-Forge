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

module core_top #(
    // 1. Global Bus and Core-to-Core Link
    parameter   GBUS_DATA   = 64,               // Data Bitwidth
    parameter   GBUS_ADDR   = 12,               // Memory Space
    
    // 2. Core Memory (WMEM and KV Cache)
    parameter   WMEM_DEPTH  = 1024,             // WMEM Size
    parameter   CACHE_DEPTH = 1024,             // KV Cache Size

    // 3. Core Buffer (LBUF and ABUF)
    parameter   LBUF_DATA   = 8*64,             // LBUF Data Bitwidth
    parameter   LBUF_DEPTH  = 16,               // LBUF Size

    // 4. Computing Logic
    parameter   MAC_NUM   = 64,                 // MAC Line Size
    parameter   IDATA_BIT = 8,                  // Input and Output Bitwidth
    parameter   ODATA_BIT = 32,                 // Partial Sum Bitwidth
    
    // 5. Config Signals
    parameter   CDATA_BIT = 8
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Global Config Signals
    input       [CDATA_BIT-1:0] cfg_acc_num,
    input       [ODATA_BIT-1:0] cfg_quant_scale,
    input       [ODATA_BIT-1:0] cfg_quant_bias,
    input       [ODATA_BIT-1:0] cfg_quant_shift,

    // Channel - Global Bus to Access Core Memory and MAC Result
    // 1. Write Channel
    //      1.1 Chip Interface -> WMEM for Weight Upload
    //      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
    //      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
    // 2. Read Channel
    //      2.1 WMEM       -> Chip Interface for Weight Check
    //      2.2 KV Cache   -> Chip Interface for KV Checnk
    //      2.3 MAC Result -> Vector Engine  for Post Processing
    input       [GBUS_ADDR-1:0] gbus_addr,
    input                       gbus_wen,
    input       [GBUS_DATA-1:0] gbus_wdata,
    input                       gbus_ren,
    output  reg [GBUS_DATA-1:0] gbus_rdata,     // To Chip Interface (Debugging) and Vector Engine (MAC)
    output  reg                 gbus_rvalid,

    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    input                       vlink_enable,
    input       [GBUS_DATA-1:0] vlink_wdata,
    input                       vlink_wen,
    output      [GBUS_DATA-1:0] vlink_rdata,
    output                      vlink_rvalid,
    // Horizontal for Activation Propagation
    // input                    vlink_enable,   // No HLING_ENABLE for Activaton
    input       [GBUS_DATA-1:0] hlink_wdata,
    input                       hlink_wen,
    output      [GBUS_DATA-1:0] hlink_rdata,
    output                      hlink_rvalid,

    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    input       [GBUS_ADDR-1:0] cmem_waddr,     // Write Value to KV Cache
    input                       cmem_wen,
    input       [GBUS_ADDR-1:0] cmem_raddr,
    input                       cmem_ren,
    // Local Buffer Access for Weight and KV Cache
    //input                     lbuf_mux,       // Annotate for Double-Buffering LBUF
    input       [LBUF_ADDR-1:0] lbuf_waddr,
    input       [LBUF_ADDR-1:0] lbuf_raddr,
    input                       lbuf_ren,
    // Local Buffer Access for Activation
    //input                     abuf_mux,
    input       [LBUF_ADDR-1:0] abuf_waddr,
    input       [LBUF_ADDR-1:0] abuf_raddr,
    input                       abuf_ren
    // MAC Output for Post-Processing in Vector Engine
    // Merged to GBUS
);

    // =============================================================================
    // Core Memory Module for Weight and KV Cache Access

    wire    [GBUS_DATA-1:0]     gbus_mem_rdata;
    wire                        gbus_mem_rvalid;

    wire    [GBUS_DATA-1:0]     cmem_wdata;
    wire                        cmem_wen_comb;

    wire    [LBUF_DATA-1;0]     lbuf_rdata;
    wire                        lbuf_rvalid;

    core_mem    #(.GBUS_DATA(GBUS_DATA), .GBUS_ADDR(GBUS_ADDR), .WMEM_DEPTH(WMEM_DEPTH),
                  .CACHE_DEPTH(CACHE_DEPTH), .LBUF_DATA(LBUF_DATA), .LBUF_DEPTH(LBUF_DEPTH)) mem_inst (
        .clk                    (clk),
        .rst                    (rst),

        .gbus_addr              (gbus_addr),
        .gbus_wen               (gbus_wen),
        .gbus_wdata             (gbus_wdata),
        .gbus_ren               (gbus_ren),
        .gbus_rdata             (gbus_mem_rdata),
        .gbus_rvalid            (gbus_mem_rvalid),

        .clink_enable           (vlink_enable),
        .clink_wdata            (vlink_wdata),
        .clink_wen              (vlink_wen),
        .clink_rdata            (vlink_rdata),
        .clink_rvalid           (vlink_rvalid),

        .cmem_waddr             (cmem_waddr),
        .cmem_wen               (cmem_wen_comb),
        .cmem_wdata             (cmem_wdata),
        .cmem_raddr             (cmem_raddr),
        .cmem_ren               (cmem_ren),

        //.lbuf_mux             (lbuf_mux),
        .lbuf_waddr             (lbuf_waddr),
        .lbuf_raddr             (lbuf_raddr),
        .lbuf_ren               (lbuf_ren),
        .lbuf_rdata             (lbuf_rdata),
        .lbuf_rvalid            (lbuf_rvalid)
    );

    // =============================================================================
    // Core Buffer Module for Activation Access

    wire    [LBUF_DATA-1:0]     abuf_rdata;
    wire                        abuf_rvalid;

    core_buf    #(.GBUS_DATA(GBUS_DATA), ABUF_DATA(LBUF_DATA), .ABUF_DEPTH(LBUF_DEPTH)) buf_inst (
        .clk                    (clk),
        .rst                    (rst),

        .clink_wdata            (hlink_wdata),
        .clink_wen              (hlink_wen),
        .clink_rdata            (hlink_rdata),
        .clink_rvalid           (hlink_rvalid),

        //.abuf_mux             (abuf_mux),
        .abuf_waddr             (abuf_waddr),
        .abuf_raddr             (abuf_raddr),
        .abuf_ren               (abuf_ren),
        .abuf_rdata             (abuf_rdata),
        .abuf_rvalid            (abuf_rvalid)
    );

    // =============================================================================
    // MAC Module

    wire    [ODATA_BIT-1:0]     mac_odata;
    wire                        mac_odata_valid;

    core_mac    #(.MAC_NUM(MAC_NUM), .IDATA_BIT(IDATA_BIT)) mac_inst (
        .clk                    (clk),
        .rst                    (rst),

        .idataA                 (abuf_rdata),
        .idataB                 (lbuf_rdata),
        .idata_valid            (abuf_rvalid && lbuf_rvalid),
        .odata                  (mac_odata),
        .odata_valid            (mac_odata_valid)
    );

    // =============================================================================
    // ACC Module

    wire    [ODATA_BIT-1:0]     acc_odata;
    wire                        acc_odata_valid;

    core_acc    #(.IDATA_BIT(ODATA_BIT), .ODATA_BIT(ODATA_BIT), .CDATA_BIT) acc_inst (
        .clk                    (clk),
        .rst                    (rst),
        .cfg_acc_num            (cfg_acc_num),

        .idata                  (mac_odata),
        .idata_valid            (mac_odata_valid),
        .odata                  (acc_odata),
        .odata_valid            (acc_odata_valid)
    );

    // =============================================================================
    // Quantization Module

    wire    [IDATA_BIT-1:0]     quant_odata;
    wire                        quant_odata_valid;

    core_quant  #(.IDATA_BIT(ODATA_BIT), .ODATA_BIT(IDATA_BIT)) quant_inst (
        .clk                    (clk),
        .rst                    (rst),

        .cfg_quant_scale        (cfg_quant_scale),
        .cfg_quant_bias         (cfg_quant_bias),
        .cfg_quant_shift        (cfg_quant_shift),

        .idata                  (acc_odata),
        .idata_valid            (acc_odata_valid),
        .odata                  (quant_odata),
        .odata_valid            (quant_odata_valid)
    );

    // =============================================================================
    // MAC Output Series to Parallel. To Match GBUS Bitwidth
    // TODO: Support byte mask in case the results can't form a complete word in the last transmission

    wire    [GBUS_DATA-1:0]     core_odata;
    wire                        core_odata_valid;

    align_s2p   #(.IDATA_BIT(IDATA_BIT), ODATA_BIT(GBUS_DATA)) mac_s2p (
        .clk                    (clk),
        .rst                    (rst),

        .idata                  (quant_odata),
        .idata_valid            (quant_odata_valid),
        .odata                  (core_odata),
        .odata_valid            (core_odata_valid)
    );

    // =============================================================================
    // Core Output Management

    // 1. Core -> KV Cache for Value
    assign  cmem_wdata    = core_odata;
    assign  cmem_wen_comb = core_odata_valid && cmem_wen;

    // 2. Core -> GBUS
    //      2.1 Weight or KV Cache -> GBUS for Debugging
    //      2.2 MAC Result         -> GBUS for Post Processing in Vector Engine
    always @(*) begin
        gbus_rvalid = gbus_mem_rvalid || core_odata_valid;
        if (gbus_mem_rvalid) begin // Read Weight and KV Cache only when GBUS_REN is set to high.
            gbus_rdata = gbus_mem_rdata;
        end
        else begin // Default: Read MAC Result when ODATA_VALID is high.
            gbus_rdata = core_odata;
        end
    end
    
endmodule