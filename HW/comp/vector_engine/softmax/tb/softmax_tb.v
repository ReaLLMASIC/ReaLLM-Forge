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

module consmax_tb ();
    parameter   IDATA_BIT = 16;  // Input Segement in INT
    parameter   ODATA_BIT = 16;  // Output Data
    parameter   CDATA_BIT = 8;  // Config Data
    parameter   LUT_ADDR = 9;
    parameter   LUT_EXP  = 8; // Exponent
    parameter   LUT_MAT  = 7;  // Mantissa
    parameter   LUT_DEPTH = 2 ** LUT_ADDR;
    parameter   LUT_DATA = 16;

    reg                       clk;
    reg                       rst;

    // Control Signals
    reg       [CDATA_BIT-1:0] cfg_consmax_shift;

    // LUT Interface
    reg       [LUT_ADDR-1:0] lut_waddr;
    reg                       lut_wen;
    reg       [LUT_DATA-1:0]  lut_wdata;

    // Data Signals
    reg       [IDATA_BIT-1:0] idata;
    reg                       idata_valid;
    reg [ODATA_BIT-1:0] odata;
    reg                 odata_valid;

    //random floating point gen
    reg  rand_sign;
    reg  [LUT_EXP-1:0] rand_exp;
    reg  [LUT_MAT-1:0] rand_man;
    
    softmax dut (.*);
    // initial begin
    //     $sdf_annotate("syn/divider.syn.sdf",divider,,,"MAXIMUM");
    // end

    always #2 clk = ~clk;

    initial begin
        $dumpfile("consmax.vcd");
        $dumpvars(0, consmax_tb) ;
        //$fsdbDumpfile("consmax.fsdb");
        //$fsdbDumpvars(0,consmax_tb);
        clk=0;
        rst=1;
        lut_wen=0;
        lut_waddr=0;
        lut_wdata=0;
        idata=0;
        idata_valid=0;
        cfg_consmax_shift=4;
        idata_valid=0;

        @(negedge clk)
        @(negedge clk)
        rst=0;

        //initializing LUT
        write_lut();

        repeat (500) begin
            input_gen();
        end

        $finish;
    end

    task write_lut;
        lut_wen=1;
        for(int i=0;i<=LUT_DEPTH;i++) begin
            @(negedge clk)
            lut_waddr=i;
            rand_sign=$random%2;
            rand_exp=$random%(2**LUT_EXP);
            rand_man=$random%(2**LUT_MAT);
            lut_wdata={rand_sign,rand_exp,rand_man};
        end
        lut_wen=0;
    endtask

    task input_gen;
        @(posedge clk)
        rand_sign=$random%2;
        rand_exp=$random%(2**LUT_EXP);
        rand_man=$random%(2**LUT_MAT);
        idata={rand_sign,rand_exp,rand_man};
        idata_valid=1;
        @(posedge clk)
        rand_sign=$random%2;
        rand_exp=$random%(2**LUT_EXP);
        rand_man=$random%(2**LUT_MAT);
        idata={rand_sign,rand_exp,rand_man};
        idata_valid=1;

    endtask
endmodule
