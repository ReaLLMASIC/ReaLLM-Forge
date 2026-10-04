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

module fmul_tb();

    localparam I_EXP        = 8;
    localparam I_MAT        = 23;
    localparam I_DATA       = I_EXP+I_MAT+1;

    localparam half_cycle   = 10;
    localparam delay        = 2;

    integer fd,fd1;
    integer i;
    reg [63:0] mem [0:2047];
    reg [63:0] bus;

    wire [31:0] result;
    fmul fmul0(
        .a_in(bus[63:32]),
        .b_in(bus[31: 0]),
        .result(result)
    );

    reg [31:0] expected_result;
    reg [31:0] temp1;
    reg [31:0] temp2;
    reg [31:0] temp3;
    initial begin

        $dumpfile("fmul.vcd");
        $dumpvars;

        $readmemh("test.mem", mem);

        fd                  = $fopen("fmul_sv_result.txt", "w"); 
        fd1                 = $fopen("fmul_cases.txt", "r"); 
        for(i               =0; i<20; i+=1) begin
            #half_cycle bus = mem[i];
            $fscanf(fd1, "%f\t%f\t%f\t%h\n", temp1, temp2, temp3,expected_result);
            $display("%f %f %f %f",temp1, temp2, temp3,expected_result);
            $fstrobe(fd, "result: %h, expected: %h", result, expected_result);
        end

        $finish;

    end


endmodule

