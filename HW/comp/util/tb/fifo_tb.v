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

module tb_fifo;

    parameter DATA_BIT = 64;
    parameter DEPTH    = 16;
    parameter ADDR_BIT = $clog2(DEPTH);

    reg wclk, wrst, wen, ren;
    reg [DATA_BIT-1:0] wdata;
    wire [DATA_BIT-1:0] rdata;
    wire wfull, rempty;

    reg clk, rst;
    reg [DATA_BIT-1:0] sync_wdata;
    wire [DATA_BIT-1:0] sync_rdata;
    wire sync_werror, sync_wfull, sync_rerror, sync_rempty;

    async_fifo #(
        .DATA_BIT(DATA_BIT),
        .DEPTH(DEPTH),
        .ADDR_BIT(ADDR_BIT)
    ) uut_async_fifo (
        .wclk(wclk),
        .wrst(wrst),
        .wen(wen),
        .wdata(wdata),
        .wfull(wfull),
        .rclk(clk),
        .rrst(rst),
        .ren(ren),
        .rdata(rdata),
        .rempty(rempty)
    );

    sync_fifo #(
        .DATA_BIT(DATA_BIT),
        .DEPTH(DEPTH),
        .ADDR_BIT(ADDR_BIT)
    ) uut_sync_fifo (
        .clk(clk),
        .rst(rst),
        .wen(wen),
        .wdata(sync_wdata),
        .werror(sync_werror),
        .wfull(sync_wfull),
        .ren(ren),
        .rdata(sync_rdata),
        .rerror(sync_rerror),
        .rempty(sync_rempty)
    );

    always #5 wclk     = ~wclk;
    always #5 clk      = ~clk;

    initial begin
        wclk           = 0;
        wrst           = 1;
        wen            = 0;
        ren            = 0;
        wdata          = 0;
        clk            = 0;
        rst            = 1;
        sync_wdata     = 0;

        #10;
        wrst           = 0;
        rst            = 0;
        #10;
        wrst           = 1;
        rst            = 1;
		
        // Write and read in async_fifo
        #10;
        wrst           = 0;
        wen            = 1;
        wdata          = 64'hAAAA_BBBB_CCCC_DDDD;
        #10;
        wen            = 0;
        ren            = 1;
        #10;
        ren            = 0;
        #20;

        // Write and read in sync_fifo
        #10;
        rst            = 0;
        wen            = 1;
        sync_wdata     = 64'h1111_2222_3333_4444;
        #10;
        wen            = 0;
        ren            = 1;
        #10;
        ren            = 0;
        #20;

        // Fill and empty the async_fifo
        #10;
        wrst           = 0;
        wen            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            wdata      = i;
            #10;
        end
        wen            = 0;
        ren            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            #10;
        end
        ren            = 0;
        #20;

        // Fill and empty the sync_fifo
        #10;
        rst            = 0;
        wen            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            sync_wdata = i;
            #10;
        end
        wen            = 0;
        ren            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            #10;
        end
        ren            = 0;
        #20;

        // Check full and empty flags in async_fifo
        #10;
        wrst           = 0;
        wen            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            wdata      = i;
            #10;
        end
        wen            = 0;
        ren            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            #10;
        end
        ren            = 0;
        #20;

        // Check full and empty flags in sync_fifo
        #10;
        rst            = 0;
        wen            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            sync_wdata = i;
            #10;
        end
        wen            = 0;
        ren            = 1;
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            #10;
        end
        ren            = 0;
        #20;

        #100;
        $stop;
    end

    initial begin
        $monitor("Time =%0t, wclk=%b, wrst=%b, wen=%b, wdata=%h, wfull=%b, clk=%b, rst=%b, ren=%b, rdata=%h, rempty=%b, sync_wdata=%h, sync_rdata=%h, sync_wfull=%b, sync_rempty=%b", 
                 $time, wclk, wrst, wen, wdata, wfull, clk, rst, ren, rdata, rempty, sync_wdata, sync_rdata, sync_wfull, sync_rempty);
    end

endmodule
