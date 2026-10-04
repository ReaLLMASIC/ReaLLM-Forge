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

module tb_align;

    parameter IDATA_BIT_S2P = 64;
    parameter ODATA_BIT_S2P = 256;
    parameter IDATA_BIT_P2S = 256;
    parameter ODATA_BIT_P2S = 64;

    reg clk;
    reg rstn;

    reg [IDATA_BIT_S2P-1:0] idata_s2p;
    reg idata_valid_s2p;
    wire [ODATA_BIT_S2P-1:0] odata_s2p;
    wire odata_valid_s2p;

    reg [IDATA_BIT_P2S-1:0] idata_p2s;
    reg idata_valid_p2s;
    wire [ODATA_BIT_P2S-1:0] odata_p2s;
    wire odata_valid_p2s;

    align_s2p #(
        .IDATA_BIT(IDATA_BIT_S2P),
        .ODATA_BIT(ODATA_BIT_S2P)
    ) align_s2p_inst (
        .clk(clk),
        .rstn(rstn),
        .idata(idata_s2p),
        .idata_valid(idata_valid_s2p),
        .odata(odata_s2p),
        .odata_valid(odata_valid_s2p)
    );

    align_p2s #(
        .IDATA_BIT(IDATA_BIT_P2S),
        .ODATA_BIT(ODATA_BIT_P2S)
    ) align_p2s_inst (
        .clk(clk),
        .rstn(rstn),
        .idata(idata_p2s),
        .idata_valid(idata_valid_p2s),
        .odata(odata_p2s),
        .odata_valid(odata_valid_p2s)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rstn = 0;
        idata_s2p = 0;
        idata_valid_s2p = 0;
        idata_p2s = 0;
        idata_valid_p2s = 0;

        #10;
        rstn = 1;

        // Normal operation for align_s2p
        #10;
        idata_valid_s2p = 1;
        idata_s2p = 64'hAAAAAAAAAAAAAAAA;
        #10;
        idata_s2p = 64'hBBBBBBBBBBBBBBBB;
        #10;
        idata_s2p = 64'hCCCCCCCCCCCCCCCC;
        #10;
        idata_s2p = 64'hDDDDDDDDDDDDDDDD;
        #10;
        idata_valid_s2p = 0;
        #20;

        // Normal operation for align_p2s
        idata_valid_p2s = 1;
        idata_p2s = 256'hAAAABBBBCCCCDDDDAAAABBBBCCCCDDDDAAAABBBBCCCCDDDDAAAABBBBCCCCDDDD;
        #10;
        idata_valid_p2s = 0;
        #20;

        // Reset
        rstn = 0;
        #10;
        rstn = 1;
        #10;

        // Edge cases
        idata_valid_s2p = 1;
        idata_s2p = 64'hFFFFFFFFFFFFFFFF;
        #10;
        idata_valid_s2p = 0;
        #30;

        idata_valid_p2s = 1;
        idata_p2s = 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        #10;
        idata_valid_p2s = 0;
        #20;

        $finish;
    end

    initial begin
        $monitor("Time: %0t, idata_s2p: %h, idata_valid_s2p: %b, odata_s2p: %h, odata_valid_s2p: %b, idata_p2s: %h, idata_valid_p2s: %b, odata_p2s: %h, odata_valid_p2s: %b", 
                 $time, idata_s2p, idata_valid_s2p, odata_s2p, odata_valid_s2p, idata_p2s, idata_valid_p2s, odata_p2s, odata_valid_p2s);
    end

endmodule
