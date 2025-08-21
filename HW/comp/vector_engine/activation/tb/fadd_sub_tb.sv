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

`timescale 1ns / 1ps

module tb_fadd_sub;

    parameter I_EXP = 8;
    parameter I_MNT = 23;
    parameter I_DATA = I_EXP + I_MNT + 1;

    reg [I_DATA-1:0] a_operand, b_operand;
    reg sub;
    wire [I_DATA-1:0] result;

    fadd_sub #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) uut (
        .a_operand(a_operand),
        .b_operand(b_operand),
        .sub(sub),
        .result(result)
    );

    initial begin
        a_operand = 32'h0;
        b_operand = 32'h0;
        sub = 0;

        $display("Test: 0 + 0");
        a_operand = 32'h00000000; // 0.0
        b_operand = 32'h00000000; // 0.0
        sub = 0;
        #10;
        $display("Expected: 0x00000000, Got: 0x%08X", result);

        $display("Test: 0 - 0");
        sub = 1;
        #10;
        $display("Expected: 0x00000000, Got: 0x%08X", result);

        $display("Test: 1.0 + 1.0");
        a_operand = 32'h3F800000; // 1.0
        b_operand = 32'h3F800000; // 1.0
        sub = 0;
        #10;
        $display("Expected: 0x40000000, Got: 0x%08X", result); // 2.0

        $display("Test: 1.0 - 1.0");
        sub = 1;
        #10;
        $display("Expected: 0x00000000, Got: 0x%08X", result); // 0.0

        $display("Test: -1.0 + 1.0");
        a_operand = 32'hBF800000; // -1.0
        b_operand = 32'h3F800000; // 1.0
        sub = 0;
        #10;
        $display("Expected: 0x00000000, Got: 0x%08X", result); // 0.0

        $display("Test: -1.0 - 1.0");
        sub = 1;
        #10;
        $display("Expected: 0xC0000000, Got: 0x%08X", result); // -2.0

        $display("Test: 2.0 + (-1.0)");
        a_operand = 32'h40000000; // 2.0
        b_operand = 32'hBF800000; // -1.0
        sub = 0;
        #10;
        $display("Expected: 0x3F800000, Got: 0x%08X", result); // 1.0

        $display("Test: 2.0 - (-1.0)");
        sub = 1;
        #10;
        $display("Expected: 0x40400000, Got: 0x%08X", result); // 3.0

        $display("Test: -2.0 + (-1.0)");
        a_operand = 32'hC0000000; // -2.0
        b_operand = 32'hBF800000; // -1.0
        sub = 0;
        #10;
        $display("Expected: 0xC0400000, Got: 0x%08X", result); // -3.0

        $finish;
    end

endmodule
