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
    parameter   GBUS_DATA = `ARR_GBUS_DATA,

    // Activation Buffer. Avoid direct access to large memory.
    parameter   ABUF_DATA  = `ARR_LBUF_DATA,                  // Data Bitwidth
    parameter   ABUF_DEPTH = `ARR_LBUF_DEPTH,                    // Depth
    parameter   ABUF_ADDR  = $clog2(ABUF_DEPTH),     // Address Bitwidth
    parameter   ALERT_DEPTH = 3
)(
    // Global Signals
    input                       clk,
    input                       rstn,

    // Channel - Core-to-Core Link
    input       [GBUS_DATA-1:0] clink_wdata,
    input                       clink_wen,
    output      [GBUS_DATA-1:0] clink_rdata,
    output                      clink_rvalid,

    // Channel - Activation Buffer for MAC Operation
    input                       abuf_ren,
    output      [ABUF_DATA-1:0] abuf_rdata,
    output  reg                 abuf_rvalid,
    output reg                  abuf_empty,
        output reg                  abuf_reuse_empty,
    output reg                  abuf_full,
    //for activation reuse
    input                       abuf_reuse_ren, //reuse pointer logic, when enable
    input                       abuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value

    output reg                  abuf_almost_full
    );

    // =============================================================================
    // Memory Instantization: Dual-Port or Double-Buffering ABUF
    // TODO: Design Exploration for DP/DB and SRAM/REGFILE/DFF ABUF
    wire    [ABUF_DATA-1:0]     abuf_wdata;
    wire    [ABUF_DATA-1:0]     abuf_rdata_mem;
    wire                        abuf_wen;
    reg    [ABUF_ADDR:0]          abuf_waddr;
    reg    [ABUF_ADDR:0]          abuf_raddr;
    reg     [ABUF_ADDR:0]       reuse_raddr; //for reuse pointer.
    reg    [ABUF_ADDR-1:0]     abuf_raddr_mux;
    assign abuf_raddr_mux= abuf_reuse_ren ? reuse_raddr[ABUF_ADDR-1:0] : abuf_raddr[ABUF_ADDR-1:0];

    abuf_64 abuf_inst(
        .ickwp0(clk),
        .iwenp0(abuf_wen),
        .iawp0(abuf_waddr[ABUF_ADDR-1:0]),
        .idinp0(abuf_wdata),
        .ickrp0(clk),
        .irenp0(abuf_ren | abuf_reuse_ren),
        .iarp0(abuf_raddr_mux),
        .iclkbyp(1'b1),
        .imce(1'b0),
        .irmce(2'b0),
        .ifuse(1'b1),
        .iwmce(4'b0),
        .odoutp0(abuf_rdata_mem)
    );

    always @(posedge clk or negedge rstn) begin
        if(~rstn)begin
            abuf_raddr <= '0;
            abuf_waddr <= '0;
            reuse_raddr <= '0;
        end else begin
            if(abuf_ren & abuf_wen) begin
                abuf_raddr <= abuf_raddr + 1;
                abuf_waddr <= abuf_waddr + 1;
                reuse_raddr<= reuse_raddr+ 1;
            end else if(abuf_ren & ~abuf_empty) begin
                abuf_raddr <= abuf_raddr + 1;
                reuse_raddr<= reuse_raddr+ 1;
            end else if(abuf_wen & ~abuf_full) begin
                abuf_waddr <= abuf_waddr + 1;
            end

            if(abuf_reuse_ren & abuf_wen & ~abuf_reuse_rst) begin
                reuse_raddr<= reuse_raddr+ 1;
                abuf_waddr <= abuf_waddr + 1;
            end else if(abuf_reuse_ren & abuf_wen & abuf_reuse_rst) begin
                reuse_raddr<= abuf_raddr;
                abuf_waddr <= abuf_waddr + 1;
            end 
            else if(abuf_reuse_ren & ~abuf_empty & ~abuf_reuse_rst) begin
                reuse_raddr<= reuse_raddr+ 1;
            end else if(abuf_reuse_ren & ~abuf_empty & abuf_reuse_rst) begin
                reuse_raddr<= abuf_raddr;
            end
        end
    end

    assign abuf_empty = (abuf_raddr == abuf_waddr);
    assign abuf_reuse_empty = (reuse_raddr == abuf_waddr);
    assign abuf_full = (abuf_raddr[ABUF_ADDR] ^ abuf_waddr[ABUF_ADDR]) & //should abuf_raddr[ABUF_ADDR] be  at [ABUF_ADDR-1] instead?
            (abuf_raddr[ABUF_ADDR-1:0] == abuf_waddr[ABUF_ADDR-1:0]);
    assign abuf_rdata = abuf_rdata_mem;

    always@(*) begin
        if(abuf_raddr[ABUF_ADDR] ^ abuf_waddr[ABUF_ADDR])
            abuf_almost_full=((abuf_raddr[ABUF_ADDR-1:0]-abuf_waddr[ABUF_ADDR-1:0])<=ALERT_DEPTH);
        else
            abuf_almost_full=((abuf_waddr[ABUF_ADDR-1:0]-abuf_raddr[ABUF_ADDR-1:0])>=ABUF_DEPTH-ALERT_DEPTH);
    end
    

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            abuf_rvalid <= 1'b0;
        end
        else begin
            abuf_rvalid <= (abuf_ren | abuf_reuse_ren) & !abuf_empty;
        end
    end

    // =============================================================================
    // Core-to-Core Link Channel

    // 1. Write Channel: CLINK -> Core
    reg     [GBUS_DATA-1:0]     clink_reg;
    reg                         clink_reg_valid;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            clink_reg <= 'd0;
        end
        else if (clink_wen) begin
            clink_reg <= clink_wdata;
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
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
        .rstn                    (rstn),
        .idata                  (clink_reg),
        .idata_valid            (clink_reg_valid),
        .odata                  (abuf_wdata),
        .odata_valid            (abuf_wen)
    );

endmodule 
