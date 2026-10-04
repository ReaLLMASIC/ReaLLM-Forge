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
// core_array module
module core_array #(
    parameter H_NUM = `ARR_HNUM,
    parameter V_NUM = `ARR_VNUM,

    parameter GBUS_DATA = `ARR_GBUS_DATA,
    parameter GBUS_ADDR = `ARR_GBUS_ADDR,

    parameter LBUF_DEPTH = `ARR_LBUF_DEPTH,
    parameter LBUF_DATA =  `ARR_LBUF_DATA,
    parameter LBUF_ADDR   = $clog2(LBUF_DEPTH),

    parameter CDATA_BIT = `ARR_CDATA_BIT,

    parameter ODATA_BIT = `ARR_ODATA_BIT,
    parameter IDATA_BIT = `ARR_IDATA_BIT,
    parameter MAC_NUM   = `ARR_MAC_NUM,

    parameter   WMEM_DEPTH  = `ARR_WMEM_DEPTH,             // WMEM Size
    parameter   CACHE_DEPTH = `ARR_CACHE_DEPTH              // KV Cache Size
    
) (
    // Global Signals
    input                       clk,
    input                       rstn,
    // Global Config Signals
    input    CFG_ARR_PACKET  arr_cfg,
    input    [H_NUM-1:0][CDATA_BIT-1:0]     cfg_acc_num,
    // Channel - Global Bus to Access Core Memory and MAC Result
    // 1. Write Channel
    //      1.1 Chip Interface -> WMEM for Weight Upload
    //      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
    //      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
    // 2. Read Channel
    //      2.1 WMEM       -> Chip Interface for Weight Check
    //      2.2 KV Cache   -> Chip Interface for KV Checnk
    //      2.3 MAC Result -> Vector Engine  for Post Processing
    input            [H_NUM-1:0][GBUS_ADDR-1:0]     gbus_addr,
    input            CTRL                           gbus_wen,
    input            [H_NUM-1:0][GBUS_DATA-1:0]     gbus_wdata,     // From Global SRAM for weight loading
    input            CTRL                           gbus_ren,
    output   logic   [H_NUM-1:0][GBUS_DATA-1:0]     gbus_rdata,     // To Chip Interface (Debugging) and Vector Engine (MAC)
    output   logic   [H_NUM-1:0] [V_NUM-1:0]        gbus_rvalid,
    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    input                                           vlink_enable,
    input            [V_NUM-1:0][GBUS_DATA-1:0]     vlink_wdata,
    input            [V_NUM-1:0]                    vlink_wen,
    output   logic   [V_NUM-1:0][GBUS_DATA-1:0]     vlink_rdata,
    output   logic   [V_NUM-1:0]                    vlink_rvalid,
    // Horizontal for Activation Propagation
    input            [H_NUM-1:0][GBUS_DATA-1:0]     hlink_wdata,    //hlink_wdata go through reg, to hlink_rdata
    input            [H_NUM-1:0]                    hlink_wen,     
    output   logic   [H_NUM-1:0][GBUS_DATA-1:0]     hlink_rdata,
    output   logic   [H_NUM-1:0]                    hlink_rvalid,
    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    input            CMEM_ARR_PACKET                arr_cmem,
    // Local Buffer Access for Weight and KV Cache
    output           CTRL        lbuf_empty,
    output           CTRL        lbuf_reuse_empty,
    input            CTRL        lbuf_reuse_ren, //reuse pointer logic, when enable
    input            CTRL        lbuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    output           CTRL        lbuf_full,
    output           CTRL        lbuf_almost_full,
    input            CTRL        lbuf_ren,
    // Local Buffer Access for Activation
    output           CTRL        abuf_empty,
    output           CTRL        abuf_reuse_empty,
    input            CTRL        abuf_reuse_ren, //reuse pointer logic, when enable
    input            CTRL        abuf_reuse_rst,  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    output           CTRL        abuf_full,
    output           CTRL        abuf_almost_full,
    input            CTRL        abuf_ren
);

//from spi
CFG_ARR_PACKET arr_cfg_reg;
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        arr_cfg_reg<='0;
    end
    else begin
        arr_cfg_reg<=arr_cfg;
    end
end

logic [H_NUM-1:0][V_NUM-1:0][CDATA_BIT-1:0] cfg_acc_num_reg;
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        cfg_acc_num_reg <= '0;
    end
    else begin
        for(int i=0; i<H_NUM; ++i) begin
            for(int j=0;j<V_NUM;++j) begin
                if(j==0) begin
                    cfg_acc_num_reg[i][0] <= cfg_acc_num[i];
                end
                else begin
                    cfg_acc_num_reg[i][j] <= cfg_acc_num_reg[i][j-1];
                end
            end
        end
    end
end


G_DATA vlink_wdata_temp;
G_DATA vlink_rdata_temp;
CTRL   vlink_wen_temp;
CTRL   vlink_rvalid_temp;

G_DATA hlink_wdata_temp;
G_DATA hlink_rdata_temp;
CTRL   hlink_wen_temp;
CTRL   hlink_rvalid_temp;

G_ADDR gbus_addr_temp;
G_DATA gbus_wdata_temp;
G_DATA gbus_rdata_temp;
always_comb begin:hlink_vlink_connection
    for(int i=0; i<H_NUM; ++i) begin
        for(int j=0;j<V_NUM;++j) begin
            if(i==0) begin 
                if(j==0) begin//(0,0)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_wdata[j];//todo 3_17 weight sharing query attention!!!
                    vlink_wen_temp[i][j]=vlink_wen[j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_wdata[i];
                    hlink_wen_temp[i][j]=hlink_wen[i];
                end
                else if(j < V_NUM-1) begin//(0,1 to V_NUM-2)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_wdata[j];  
                    vlink_wen_temp[i][j]=vlink_wen[j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_rdata_temp[i][j-1];
                    hlink_wen_temp[i][j]=hlink_rvalid_temp[i][j-1];
                end
                else begin//(0,V_NUM-1)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_wdata[j];
                    vlink_wen_temp[i][j]=vlink_wen[j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_rdata_temp[i][j-1];
                    hlink_wen_temp[i][j]=hlink_rvalid_temp[i][j-1];
                    hlink_rdata[i]=hlink_rdata_temp[i][j];
                    hlink_rvalid[i]=hlink_rvalid_temp[i][j];
                end
            end
            else if(i < V_NUM-1) begin
                if(j==0) begin //(1 to H_NUM-2,0)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_rdata_temp[i-1][j];
                    vlink_wen_temp[i][j]=vlink_rvalid_temp[i-1][j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_wdata[i];
                    hlink_wen_temp[i][j]=hlink_wen[i];
                end
                else if(j<V_NUM-1) begin //(1 to H_NUM-2,1 to V_NUM-2)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_rdata_temp[i-1][j];  
                    vlink_wen_temp[i][j]=vlink_rvalid_temp[i-1][j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_rdata_temp[i][j-1];
                    hlink_wen_temp[i][j]=hlink_rvalid_temp[i][j-1];
                end
                else begin //(1 to H_NUM-2,V_NUM-1)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_rdata_temp[i-1][j];
                    vlink_wen_temp[i][j]=vlink_rvalid_temp[i-1][j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_rdata_temp[i][j-1];
                    hlink_wen_temp[i][j]=hlink_rvalid_temp[i][j-1];
                    hlink_rdata[i]=hlink_rdata_temp[i][j];
                    hlink_rvalid[i]=hlink_rvalid_temp[i][j];
                end
            end
            else begin 
                if(j==0)begin //(H_NUM-1,0)
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_rdata_temp[i-1][j];
                    vlink_wen_temp[i][j]=vlink_rvalid_temp[i-1][j];
                    vlink_rdata[j]=vlink_rdata_temp[i][j];
                    vlink_rvalid[j]=vlink_rvalid_temp[i][j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_wdata[i];
                    hlink_wen_temp[i][j]=hlink_wen[i];
                end
                else if(j<V_NUM-1) begin
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_rdata_temp[i-1][j];  
                    vlink_wen_temp[i][j]=vlink_rvalid_temp[i-1][j];
                    vlink_rdata[j]=vlink_rdata_temp[i][j];
                    vlink_rvalid[j]=vlink_rvalid_temp[i][j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_rdata_temp[i][j-1];
                    hlink_wen_temp[i][j]=hlink_rvalid_temp[i][j-1];
                end
                else begin
                    //vlink
                    vlink_wdata_temp[i][j]=vlink_rdata_temp[i-1][j];  
                    vlink_wen_temp[i][j]=vlink_rvalid_temp[i-1][j];
                    vlink_rdata[j]=vlink_rdata_temp[i][j];
                    vlink_rvalid[j]=vlink_rvalid_temp[i][j];
                    //hlink
                    hlink_wdata_temp[i][j]=hlink_rdata_temp[i][j-1];
                    hlink_wen_temp[i][j]=hlink_rvalid_temp[i][j-1];
                    hlink_rdata[i]=hlink_rdata_temp[i][j];
                    hlink_rvalid[i]=hlink_rvalid_temp[i][j];
                end
            end
        end
    end
end
//gbus wen gbus ren high at same time, gbus_addr
always_comb begin: gbus_connection
    gbus_rdata='0;
    //initialize
    for(int i=0;i< H_NUM; ++i) begin
        for(int j=0;j< V_NUM; ++j) begin
            gbus_wdata_temp[i][j]='0;
            gbus_addr_temp[i][j]='0;
        end
    end
    //gbus_rdata
    for(int i=0;i< H_NUM; ++i) begin
        for(int j=0;j< V_NUM; ++j) begin
            if(gbus_rvalid[i][j]) begin
                gbus_rdata[i]=gbus_rdata_temp[i][j];
                break;
            end
        end
    end
    //gbus_wdata
    for(int i=0;i< H_NUM; ++i) begin
        for(int j=0;j< V_NUM; ++j) begin
            if(gbus_wen[i][j]) begin
                gbus_wdata_temp[i][j]=gbus_wdata[i];
                break;
            end
        end
    end
    //gbus_addr
    for(int i=0;i< H_NUM; ++i) begin
        for(int j=0;j< V_NUM; ++j) begin
            if(gbus_wen[i][j] | gbus_ren[i][j]) begin
                gbus_addr_temp[i][j]=gbus_addr[i];
                break;
            end
        end
    end
end

generate
    for (genvar i = 0; i < H_NUM; ++i)begin : gen_row
        for (genvar j = 0; j < V_NUM; ++j)begin : gen_col
            core_top #(
                        .GBUS_DATA  (GBUS_DATA),
                        .GBUS_ADDR  (GBUS_ADDR),
                        .WMEM_DEPTH (WMEM_DEPTH),
                        .CACHE_DEPTH (CACHE_DEPTH),
                        .LBUF_DATA   (LBUF_DATA),
                        .LBUF_DEPTH  (LBUF_DEPTH),
                        .LBUF_ADDR   (LBUF_ADDR),
                        .MAC_NUM     (MAC_NUM),
                        .IDATA_BIT   (IDATA_BIT),
                        .ODATA_BIT   (ODATA_BIT),
                        .CDATA_BIT   (CDATA_BIT)
                    ) core_top_instance (
                        .clk           (clk),
                        .rstn          (rstn),
                        .cfg_acc_num   (cfg_acc_num_reg[i][j]),
                        .cfg_quant_scale (arr_cfg_reg.cfg_quant_scale),
                        .cfg_quant_bias (arr_cfg_reg.cfg_quant_bias),
                        .cfg_quant_shift (arr_cfg_reg.cfg_quant_shift),
                        .gbus_addr     (gbus_addr_temp[i][j]), //gbus to weight mem, gbus to kv cache
                        .gbus_wen      (gbus_wen[i][j]),
                        .gbus_wdata    (gbus_wdata_temp[i][j]),
                        .gbus_ren      (gbus_ren[i][j]),
                        .gbus_rdata    (gbus_rdata_temp[i][j]),
                        .gbus_rvalid   (gbus_rvalid[i][j]),
                        .vlink_enable  (vlink_enable),
                        .vlink_wdata   (vlink_wdata_temp[i][j]), // access lbuf
                        .vlink_wen     (vlink_wen_temp[i][j]),
                        .vlink_rdata   (vlink_rdata_temp[i][j]),
                        .vlink_rvalid  (vlink_rvalid_temp[i][j]),
                        .hlink_wdata   (hlink_wdata_temp[i][j]), // access abuf
                        .hlink_wen     (hlink_wen_temp[i][j]),
                        .hlink_rdata   (hlink_rdata_temp[i][j]),
                        .hlink_rvalid  (hlink_rvalid_temp[i][j]),
                        .cmem_waddr    (arr_cmem.cmem_waddr[i][j]),
                        .cmem_wen      (arr_cmem.cmem_wen[i][j]), //cmem_wen control, when high, mac output will send to cmem, if gbus_ren is not high previous cycle, mac output will send to gbus too.
                        .cmem_raddr    (arr_cmem.cmem_raddr[i][j]),
                        .cmem_ren      (arr_cmem.cmem_ren[i][j]),
                        .lbuf_full     (lbuf_full[i][j]),
                        .lbuf_almost_full(lbuf_almost_full[i][j]),
                        .lbuf_empty    (lbuf_empty[i][j]),
                        .lbuf_reuse_empty(lbuf_reuse_empty[i][j]),
                        .lbuf_ren      (lbuf_ren[i][j]),
                        .lbuf_reuse_ren(lbuf_reuse_ren[i][j]),
                        .lbuf_reuse_rst(lbuf_reuse_rst[i][j]),
                        .abuf_full     (abuf_full[i][j]),
                        .abuf_almost_full(abuf_almost_full[i][j]),
                        .abuf_empty    (abuf_empty[i][j]),
                        .abuf_reuse_empty(abuf_reuse_empty[i][j]),
                        .abuf_reuse_ren(abuf_reuse_ren[i][j]),
                        .abuf_reuse_rst(abuf_reuse_rst[i][j]),
                        .abuf_ren      (abuf_ren[i][j])
                    );
        end
    end
endgenerate
    
endmodule
