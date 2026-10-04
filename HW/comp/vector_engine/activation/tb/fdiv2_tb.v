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

module tb_fdiv2;

    parameter I_EXP   = 8;
    parameter I_MNT   = 7;
    parameter I_DATA  = I_EXP + I_MNT + 1;

    reg [I_DATA-1:0] if32;
    wire [I_DATA-1:0] of32;

    fdiv2 #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) dut (
        .if32(if32),
        .of32(of32)
    );

    initial begin
        if32 = 0;

        // Exponent is all zeros
        if32 = {1'b0, {I_EXP{1'b0}}, {I_MNT{1'b1}}}; 
        #10;
      	$display("Exponent is all zeros");
      	$display("expected: 0 00000000 1111111 -> 0 00000000 0111111");
      	$display("%b -> %b", if32, of32);
      	$display("");

        // Exponent is 1 (special case)
        if32 = {1'b0, {{(I_EXP-1){1'b0}}, 1'b1}, {I_MNT{1'b1}}}; 
        #10;
      	$display("Exponent is 1 (special case)");
      	$display("expected: 0 0000001 1111111 -> 0 00000000 1111111");
        $display("%b -> %b", if32, of32);
      	$display("");

        // Normal exponent
        if32 = {1'b0, {4'b0010, 4'b0001}, {I_MNT{1'b1}}}; 
        #10;
      	$display("Normal exponent");
      $display("expected: 0 00100001 1111111 -> 0 00100000 1111111");
        $display("%b -> %b", if32, of32);
      	$display("");

        // Normal exponent 2
        if32 = {1'b0, {4'b0111, 4'b1000}, {I_MNT{1'b1}}}; 
        #10;
      	$display("Normal exponent 2");
      	$display("expected: 0 01111000 1111111 -> 0 01110111 1111111");
        $display("%b -> %b", if32, of32);
      	$display("");

        // Max exponent value
        if32 = {1'b0, {I_EXP{1'b1}}, {I_MNT{1'b1}}}; 
        #10;
      	$display("Max exponent value");
        $display("expected: 0 11111111 1111111 -> 0 11111110 1111111");
        $display("%b -> %b", if32, of32);
      	$display("");

        // Min exponent with sign bit 1
        if32 = {1'b1, {I_EXP{1'b0}}, {I_MNT{1'b1}}}; 
        #10;
      	$display("Min exponent with sign bit 1");
        $display("expected: 1 00000000 1111111 -> 1 00000000 0111111");
        $display("%b -> %b", if32, of32);
      	$display("");

        // Exponent is 1 (special case) with sign bit 1
        if32 = {1'b1, {{(I_EXP-1){1'b0}}, 1'b1}, {I_MNT{1'b1}}};
        #10;
       	$display("Exponent is 1 (special case) with sign bit 1");
        $display("expected: 1 00000001 1111111 -> 1 00000000 1111111");
        $display("%b -> %b", if32, of32);
    	$display("");

        // Normal exponent with sign bit 1
        if32 = {1'b1, {4'b0010, 4'b0001}, {I_MNT{1'b1}}}; 
        #10;
        $display("expected: 1 00100001 1111111 -> 1 00100000 1111111");
        $display("%b -> %b", if32, of32);
      	$display("");

        $finish;
    end

endmodule
