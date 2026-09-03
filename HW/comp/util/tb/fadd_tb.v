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

module tb_fadd;

  localparam BIT_W = 32; 
  localparam EXP_W = 8;
  localparam M_W = 23;
  
  reg [BIT_W-1:0] a_operand;
  reg [BIT_W-1:0] b_operand;

  wire [BIT_W-1:0] result;

  fadd add (
    .a_operand(a_operand),
    .b_operand(b_operand),
    .result(result)
  );

  initial begin
    a_operand = 0;
    b_operand = 0;

    // normal addition
    a_operand = 32'h4048F5C3; // 3.14
    b_operand = 32'h3FC00000; // 1.5
    #10;
    $display("3.14 + 1.5 = %h (Expected: 40947AE1)", result);
    $display("");

    // addition with zero
    a_operand = 32'h00000000; // 0.0
    b_operand = 32'h4048F5C3; // 3.14
    #10;
    $display("0.0 + 3.14 = %h (Expected: 4048F5C3)", result);
    $display("");

    // addition with two negative numbers
    a_operand = 32'hC048F5C3; // -3.14
    b_operand = 32'hBFC00000; // -1.5
    #10;
    $display("-3.14 + -1.5 = %h (Expected: C0947AE1)", result);
    $display("");

    // addition with large exponents
    a_operand = 32'h7F7FFFFF; // Largest positive normal number
    b_operand = 32'h00800000; // Smallest positive normal number
    #10;
    $display("Largest + Smallest = %h (Expected: 7F7FFFFF)", result);
    $display("");
    
    // addition with infinity
    a_operand = 32'h7F800000; // +Infinity
    b_operand = 32'h7F800000; // +Infinity
    #10;
    $display("+Infinity + +Infinity = %h (Exception Expected: 00000000)", result);
    $display("");

    // addition with -Infinity
    a_operand = 32'hFF800000; // -Infinity
    b_operand = 32'h4048F5C3; // 3.14
    #10;
    $display("-Infinity + 3.14 = %h (Exception Expected: 00000000)", result);
    $display("");

    // addition with NaN
    a_operand = 32'h7FC00000; // NaN
    b_operand = 32'h4048F5C3; // 3.14
    #10;
    $display("NaN + 3.14 = %h (Exception Expected: 00000000)", result);
    $display("");
    
    // addition resulting in a subnormal number
    a_operand = 32'h00000001; // Smallest positive subnormal number
    b_operand = 32'h00000001; // Smallest positive subnormal number
    #20;
    $display("Subnormal + Subnormal = %h (Expected: 00000002)", result);
    $display("");
    
    $finish;
  end

  initial begin
    $monitor("a_operand = %h, b_operand = %h",
             a_operand, b_operand);
  end

endmodule
