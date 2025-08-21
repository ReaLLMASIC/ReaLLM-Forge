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

module core_mem #(
    // Global Bus and Core-to-Core Link (Access WMEM and KV-Cache)
    parameter   GBUS_DATA = 64,                     // Data Bitwidth
    parameter   GBUS_ADDR = 12,                     // Address Bitwidth

    // Single-Port Weight Memory (SRAM). Config the default setting for a real application.
    // The WMEM has the same data bitwidth with GBUS.
    parameter   WMEM_DEPTH = 1024,                  // Depth
    parameter   WMEM_ADDR  = $clog2(WMEM_DEPTH),    // Address Bitwidth

    // Single-Port KV Cache (SRAM). Config the default setting for a real application.
    // The KV Cache has the same data bitwith with GBUS.
    parameter   CACHE_DEPTH = 1024,                 // Depth
    parameter   CACHE_ADDR  = $clog2(CACHE_ADDR),   // Address Bitwidth

    // Local Buffer (LBUF), SRAM or DFFs. Avoid direct access to large memory.
    parameter   LBUF_DATA  = 8*64,                  // Data Bitwidth
    parameter   LBUF_DEPTH = 16,                    // Depth
    parameter   LBUF_ADDR  = $clog2(LBUF_DEPTH)     // Address Bitwidth

)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Channel - Global Bus (GBUS) to Access Core Memory (WMEM and KV Cache)
    // Data Upload from (1) Chip Interface and (2) Vector Engine
    input       [GBUS_ADDR-1:0] gbus_addr,
    input                       gbus_wen,
    input       [GBUS_DATA-1:0] gbus_wdata,
    input                       gbus_ren,
    output  reg [GBUS_DATA-1:0] gbus_rdata,
    output  reg                 gbus_rvalid,

    // Channel - Core-to-Core Link (CLINK)
    // Global Config Signals
    input                       clink_enable,
    // No WADDR. The data from neighboring core is sent to MAC directly.
    input       [GBUS_DATA-1:0] clink_wdata,
    input                       clink_wen,
    // No RADDR. The data to neighboring core is the same as the current operand.
    output  reg [GBUS_DATA-1:0] clink_rdata,
    output  reg                 clink_rvalid,

    // Channel - Core Memory for MAC Operation
    // Write back results from MAC directly.
    input       [GBUS_ADDR-1:0] cmem_waddr,         // Assume WMEM_ADDR >= CACHE_ADDR
    input                       cmem_wen,
    input       [GBUS_DATA-1:0] cmem_wdata,
    // Read data to LBUF. No RDATA, Sent data to LBUF directly in this module.
    input       [GBUS_ADDR-1:0] cmem_raddr,
    input                       cmem_ren,

    // Channel - Local Memory (LBUF) for MAC Operation
    //input                     lbuf_mux,           // Annotate for Double-Buffering LBUF
    input       [LBUF_ADDR-1:0] lbuf_waddr,         // No WDATA or WEN. Receive data from CMEM directly in this moduel.
    input       [LBUF_ADDR-1:0] lbuf_raddr,
    input                       lbuf_ren,
    output      [LBUF_DATA-1:0] lbuf_rdata,         // To MAC
    output  reg                 lbuf_rvalid
);

    // =============================================================================
    // Memory Instantization

    // 1. Single-Port Weight Memory
    reg     [WMEM_ADDR-1:0]     wmem_addr;
    reg                         wmem_wen;
    reg     [GBUS_DATA-1:0]     wmem_wdata;
    reg                         wmem_ren;
    wire    [GBUS_DATA-1:0]     wmem_rdata;
    reg                         wmem_rvalid;

    mem_sp #(.DATA_BIT(GBUS_DATA), .DEPTH(WMEM_DEPTH))  wmem_inst (
        .clk                    (clk),
        .addr                   (wmem_addr),
        .wen                    (wmem_wen),
        .wdata                  (wmem_wdata),
        .ren                    (wmem_ren),
        .rdata                  (wmem_rdata)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wmem_rvalid <= 1'b0;
        end
        else begin
            wmem_rvalid <= wmem_ren;
        end
    end

    // 2. Single-Port KV Cache
    reg     [CACHE_ADDR-1:0]    cache_addr;
    reg                         cache_wen;
    reg     [GBUS_DATA-1:0]     cache_wdata;
    reg                         cache_ren;
    wire    [GBUS_DATA-1:0]     cache_rdata;
    reg                         cache_rvalid;

    mem_sp  #(.DATA_BIT(GBUS_DATA), .DEPTH(CACHE_DEPTH)) cache_inst (
        .clk                    (clk),
        .addr                   (cache_addr),
        .wen                    (cache_wen),
        .wdata                  (cache_wdata),
        .ren                    (cache_ren),
        .rdata                  (cache_rdata)
    );

    always @(posedge clk or posedge rst) begin

    end

    // 3. Dual-Port or Double-Buffering LBUF
    // TODO: Design Exploration for DP/DB and SRAM/REGFILE/DFF.
    wire    [LBUF_DATA-1:0]     lbuf_wdata;     // Series_to_Parallel -> LBUF_WDATA
    wire                        lbuf_wen;

    mem_dp  #(.DATA_BIT(LBUF_DATA), .DEPTH(LBUF_DEPTH)) lbuf_inst (
        .clk                    (clk),
        .waddr                  (lbuf_waddr),
        .wen                    (lbuf_wen),
        .wdata                  (lbuf_wdata),
        .raddr                  (lbuf_raddr),
        .ren                    (lbuf_ren),
        .rdata                  (lbuf_rdata)
    );

    /*
    mem_db  #(.DATA_BIT(LBUF_BIT), .DEPTH(LBUF_DEPTH))  lbuf_inst (
        .clk                    (clk),
        .sw                     (lbuf_mux),
        .waddr                  (lbuf_waddr),
        .wen                    (lbuf_wen),
        .wdata                  (lbuf_wdata),
        .raddr                  (lbuf_raddr),
        .ren                    (lbuf_ren),
        .rdata                  (lbuf_rdata)
    );
    */

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            lbuf_rvalid <= 1'b0;
        end
        else begin
            lbuf_rvalid <= lbuf_ren;
        end
    end

    // =============================================================================
    // Weight Memory Interface

    reg     [WMEM_ADDR-1:0]     wmem_waddr;
    reg     [WMEM_ADDR-1:0]     wmem_raddr;

    // 1. Write Channel:
    //      1.1 GBUS -> WMEM for Weight Loading
    always (*) begin
        wmem_waddr = gbus_addr[WMEM_ADDR-1:0];
        wmem_wen   = gbus_wen && ~gbus_addr[GBUS_ADDR-1];
        wmem_wdata = gbus_wdata;
    end

    // 2. Read Channel:
    //      2.1 WMEM -> GBUS  for Weight Check (Debugging)
    //      2.2 WMEM -> MAC   for on-core AxW
    //      2.3 WMEM -> CLINK for off-core AxW (Solved in CLINK Bundle)
    always @(*) begin
        wmem_ren = (gbus_ren && ~gbus_addr[GBUS_ADDR-1]) || (cmem_ren && ~cmem_raddr[GBUS_ADDR-1]);
        if (gbus_ren && ~gbus_addr[GBUS_ADDR-1]) begin // WMEM -> GBUS
            wmem_raddr = gbus_addr[WMEM_ADDR-1:0];
        end
        else begin // WMEM -> MAC
            wmem_raddr = cmem_raddr[WMEM_ADDR-1:0];
        end
    end

    // 3. WMEM_ADDR W/R Selection
    always @(*) begin
        if (wmem_wen) begin // Write
            wmem_addr = wmem_waddr;
        end
        else begin // Read
            wmem_addr = wmem_raddr;
        end
    end

    // =============================================================================
    // KV Cache Interface

    reg     [CACHE_ADDR-1:0]    cache_waddr;
    reg     [CACHE_ADDR-1:0]    cache_raddr;

    // 1. Write Channel:
    //      1.1 GBUS -> KV Cache
    //      1.2 Value Vector from MAC -> KV Cache
    always @(*) begin
        cache_wen = (gbus_wen && gbus_addr[GBUS_ADDR-1]) || (cmem_wen && cmem_waddr[GBUS_ADDR-1]);
        if (cmem_wen && cmem_waddr[GBUS_ADDR-1]) begin // GBUS -> KV Cache
            cache_waddr = cmem_waddr[CACHE_ADDR-1:0];
            cache_wdata = cmem_wdata;
        end
        else begin // MAC -> KV Cache
            cache_waddr = gbus_addr[CACHE_ADDR-1:0];
            cache_wdata = gbus_wdata;
        end
    end

    // 2. Read Channel:
    //      2.1 KV Cache -> GBUS  for Key/Value Check (Debugging)
    //      2.2 KV Cache -> MAC   for On-Core QxK/PxV
    //      2.3 KV Cache -> CLINK for Off-Core QxK/PxV (Solved in CLINK Bundle)
    always @(*) begin
        cache_ren = (gbus_ren && gbus_addr[GBUS_ADDR-1]) || (cmem_ren && cmem_raddr[GBUS_ADDR-1]);
        if (gbus_ren && gbus_addr[GBUS_ADDR-1]) begin // KV Cache -> GBUS
            cache_raddr = gbus_addr[CACHE_ADDR-1:0];
        end
        else begin // KV Cache -> MAC
            cache_raddr = cmem_raddr[CACHE_ADDR-1:0];
        end
    end

    // 3. CACHE_ADDR W/R Selection
    always @(*) begin
        if (cache_wen) begin // Write
            cache_addr = cache_waddr;
        end
        else begin // Read
            cache_addr = cache_raddr;
        end
    end

    // =============================================================================
    // GBUS Read Channel

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            gbus_rvalid <= 1'b0;
        end
        else begin
            gbus_rvalid <= gbus_ren;
        end
    end

    always @(*) begin
        if (gbus_rvalid && cache_rvalid) begin // KV Cache -> GBUS
            gbus_rdata = cache_rdata;
        end
        else begin // WMEM -> GBUS
            gbus_rdata = wmem_rdata;
        end
    end

    // =============================================================================
    // Core-to-Core Link Channel

    // 1. Write Channel: CLINK -> Core
    reg     [GBUS_DATA-1:0] clink_reg;
    reg                     clink_reg_valid;

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

    // 2. Read Channel: WMEM/Cache/CLINK -> CLINK
    reg                     cmem_rvalid;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cmem_rvalid <= 1'b0;
        end
        else begin
            cmem_rvalid <= cmem_ren;
        end
    end

    always @(*) begin
        if (clink_enable) begin // Enable CLINK
            clink_rvalid = clink_reg_valid || cmem_rvalid;
        end
        else begin // Disable CLINK
            clink_rvalid = 1'b0;
        end
    end

    always @(*) begin
        if (clink_reg_valid) begin // Core-(i) CLINK -> Core-(i+1) CLINK
            clink_rdata = clink_reg;
        end
        else if (cache_rvalid && cmem_rvalid) begin // Core-(i) KV Cache -> CLINK for Core-(i+1)
            clink_rdata = cache_rdata;
        end
        else begin // Core-(i) WMEM -> CLINK for Core-(i+1)
            clink_rdata = wmem_rdata;
        end
    end

    // =============================================================================
    // LBUF Write Channel: Series to Parallel

    align_s2p   #(.IDATA_BIT(GBUS_DATA), .ODATA_BIT(LBUF_DATA)) lbuf_s2p (
        .clk                    (clk),
        .rst                    (rst),
        .idata                  (clink_rdata),
        .idata_valid            (clink_rvalid || cmem_rvalid),
        .odata                  (lbuf_wdata),
        .odata_valid            (lbuf_wen)
    );

endmodule