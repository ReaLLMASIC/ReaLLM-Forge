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

module core_tb # (parameter GBUS_DATA = 64,
                  parameter GBUS_ADDR = 12,
                  parameter WMEM_DEPTH = 1024,
                  parameter CACHE_DEPTH = 1024,
                  parameter LBUF_DATA = 8*64,
                  parameter LBUF_DEPTH = 16,
                  parameter MAC_NUM = 64,
                  parameter IDATA_BIT = 8,
                  parameter ODATA_BIT = 32,
                  parameter CDATA_BIT = 8,
                  parameter LBUF_ADDR = $clog2(LBUF_DEPTH)
                           ); //LBUF = $clog2(LBUF_DEPTH)
//Global signals
logic clk;
logic rstn;
//Global Config Signals
logic [CDATA_BIT-1:0] cfg_acc_num;
logic [ODATA_BIT-1:0] cfg_quant_scale;
logic [ODATA_BIT-1:0] cfg_quant_bias;
logic [ODATA_BIT-1:0] cfg_quant_shift;

// Channel - Global Bus to Access Core Memory and MAC Result
logic [GBUS_ADDR-1:0] gbus_addr;
logic [GBUS_ADDR-1:0] gbus_addr_in_task;
logic [GBUS_ADDR-1:0] gbus_addr_in_task_2;
logic gbus_wen;
logic [GBUS_DATA-1:0] gbus_wdata;
logic gbus_ren;
logic [GBUS_DATA-1:0] gbus_rdata; //output
logic gbus_rvalid; //output

// Channel - Core-to-Core Link
// Vertical for Weight and Key/Value Propagation
logic vlink_enable;
logic [GBUS_DATA-1:0] vlink_wdata;
logic vlink_wen;
logic [GBUS_DATA-1:0] vlink_rdata; //output
logic vlink_rvalid; //output

// Horizontal for Activation Propagation
logic [GBUS_DATA-1:0] hlink_wdata;
logic hlink_wen;
logic [GBUS_DATA-1:0] hlink_rdata; //output
logic hlink_rvalid; //output

// Channel - MAC Operation
// Core Memory Access for Weight and KV Cache
logic [GBUS_ADDR-1:0] cmem_waddr;
logic cmem_wen;
logic [GBUS_ADDR-1:0] cmem_raddr;
logic cmem_ren;

// Local Buffer Access for Weight and KV Cache
logic [LBUF_ADDR-1:0] lbuf_waddr;
logic [LBUF_ADDR-1:0] lbuf_raddr;
logic lbuf_ren;

// Local Buffer Access for Activation
logic [LBUF_ADDR-1:0] abuf_waddr;
logic [LBUF_ADDR-1:0] abuf_raddr;
logic abuf_ren;

logic lbuf_empty;
logic lbuf_full;
logic abuf_empty;
logic abuf_full;

core_top #(.GBUS_DATA(GBUS_DATA), .GBUS_ADDR(GBUS_ADDR), .WMEM_DEPTH(WMEM_DEPTH), .CACHE_DEPTH(CACHE_DEPTH),
           .LBUF_DATA(LBUF_DATA), .LBUF_DEPTH(LBUF_DEPTH), .MAC_NUM(MAC_NUM),.IDATA_BIT(IDATA_BIT),
           .ODATA_BIT(ODATA_BIT), .CDATA_BIT(CDATA_BIT), .LBUF_ADDR(LBUF_ADDR)) core_inst(.*);
always begin
    #0.5 clk = ~clk;
end

task r_w_wmem(input [GBUS_ADDR-1:0]gbus_addr_in, input [3:0]read, input [3:0]cmem);

    hlink_wen = 1'b0;
    cmem_ren = 1'b0;
    vlink_wen = 1'b0;
    vlink_enable = 1'b0;
    gbus_addr = gbus_addr_in;
    gbus_wen = 1'b1;
    gbus_wdata = 1000;
    gbus_ren = 1'b0;
    @(posedge clk);
    if(read[1] & ~read[0] & ~read[2] & ~read[3]) begin
        lbuf_ren = 1'b1;
        abuf_ren = 1'b1;
    end
    else begin
        lbuf_ren = 1'b0;
        abuf_ren = 1'b0;
    end
    @(negedge clk);
    if(read == 7 && cmem == 9) begin
        cmem_wen = 1'b1;
        cmem_waddr = 64'hFFFF_FFFF_FFFF_FFFF;
    end
    else begin
        cmem_wen = 1'b0;
    end

    //lbuf_ren = 1'b0;
    //abuf_ren = 1'b0;
    gbus_ren = 1'b1;
    gbus_wen = 1'b0;
    hlink_wdata = 1200;
    hlink_wen = 1;    
    //@(negedge clk);
    cmem_ren = 1'b1;
    vlink_wen = 1'b0;
    vlink_enable = 1'b1;
   
    @(negedge clk);
endtask
logic [3:0]read;
task lbuf_abuf_fill(input [GBUS_ADDR-1:0]gbus_addr_in, input [3:0]cmem);
    gbus_addr_in_task = gbus_addr_in;
    read = 1;
    repeat(8) begin
        r_w_wmem(gbus_addr_in_task, read, cmem);
        gbus_addr_in_task = gbus_addr_in_task + 1;
        read = read + 1;
    end
    vlink_wen = 1'b0;
    //lbuf_ren = 1'b1;
    cmem_ren = 1'b0;


    //abuf_ren = 1'b1;
    hlink_wen = 1'b0;


endtask
logic [3:0] cmem;
initial begin
    $dumpfile("core.dump");
    $dumpvars(0, core_tb);
    clk = 0;
    rstn = 0;
    @(negedge clk);
    @(negedge clk);
    rstn = 1;
    //Config stuff
    cfg_acc_num = 1;
    cfg_quant_scale = 16;
    cfg_quant_bias = 10;
    cfg_quant_shift = 2;
    gbus_addr_in_task_2 = 'd0;
    cmem = 1;
    repeat(9) begin
        lbuf_abuf_fill(gbus_addr_in_task_2, cmem);
        gbus_addr_in_task_2 = gbus_addr_in_task_2 + 8;
        cmem = cmem+1;
    end
    cmem_ren = 1'b1;
    cmem_raddr = 64'hFFFF_FFFF_FFFF_FFFF;
    vlink_wen = 1'b0;
    vlink_enable = 1'b0;
    hlink_wen = 1'b0;
    //cmem_wen = 1'b1;
    //cmem_waddr = 64'hFFFF_FFFF_FFFF_FFFF;
/*
    //Global bus stuff
    gbus_addr = 'd0;
    gbus_wen = 1'b1;
    gbus_wdata = 1000;
    gbus_ren = 1'b0; //was 0

    @(negedge clk);
    gbus_ren = 1'b1;
    gbus_wen = 1'b0;
    @(negedge clk);
    gbus_ren = 1'b0;
    gbus_wen = 1'b1;
    gbus_addr = 'd1;
    @(negedge clk);
    gbus_ren = 1'b1;
    gbus_Wen = 1'b0;
    @(negedge clk);
    gbus_addr = 'd2;
    gbus_wen = 1'b1;
    @(negedge clk);
    gbus_ren = 1'b1;
    gbus_wen = 1'b0;
    @(negedge clk);
    gbus_addr = 'd3;
    gbus_wen = 1'b1;
    @(negedge clk);
    gbus_addr = 'd4;
    gbus_wdata = 1004;
    @(negedge clk);
    gbus_addr = 'd5;
    gbus_wdata = 1005;
    @(negedge clk);
    gbus_addr = 'd6;
    gbus_wdata = 1006;
    @(negedge clk);
    gbus_addr = 'd7;
    gbus_wdata = 1007;
    @(negedge clk);

    
    gbus_ren = 1'b1;
    //cmem_ren = 1'b1;
    @(negedge clk);
    cmem_ren = 1'b1;
    vlink_wen = 1'b0;
    vlink_enable = 1'b1;

    //Horizontal link
    hlink_wdata = 1200;
    hlink_wen = 1;
    //Got through here
    repeat(10) begin
        @(negedge clk);
    end
    vlink_wen = 1'b0;
    lbuf_ren = 1'b1;
    cmem_ren = 1'b0;


    abuf_ren = 1'b1;
    hlink_wen = 1'b0;
    @(negedge clk);
    //@(negedge clk);
    //lbuf_ren = 1'b0;
    //abuf_ren = 1'b0;
    @(negedge clk);
    //lbuf_ren = 1'b1;
    //@(negedge clk);
    //lbuf_ren = 1'b0;


    repeat (10) begin
        @(negedge clk);
    end

    cmem_wen = 1'b1;
    cmem_waddr = 64'hFFFF_FFFF_FFFF_FFFF;

    //Look at the KV cache output
*/

    repeat(50) begin
        @(negedge clk);
    end
    $display("Finish successfully");
    $finish;



   /* 
   //Vertical link
    vlink_enable = 1;
    vlink_wdata = 10;
    vlink_wen = 1;



    //Core mem access for Weight and KV Cache
    lbuf_waddr = 32;
    lbuf_raddr = 64;
    lbuf_ren = 1;

    //Local Buffer Access for Activation
    abuf_waddr = 16;
    abuf_waddr = 5;
    abuf_ren = 0;
    */



end



endmodule
