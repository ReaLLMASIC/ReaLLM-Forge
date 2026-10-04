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

module tb_fmul;

  localparam BIT_W = 32;
  localparam EXP_W = 8;
  localparam M_W = 23;
  localparam MULT_W = M_W + M_W + 2;

  reg [BIT_W-1:0] a_in;
  reg [BIT_W-1:0] b_in;
  wire [BIT_W-1:0] result;

  fmul mul (
    .a_in(a_in),
    .b_in(b_in),
    .result(result)
  );

  initial begin
    a_in = 0;
    b_in = 0;
    #10;

    // 1.0 * 2.0 (Expected result: 2.0)
    a_in = 32'h3F800000;  // 1.0
    b_in = 32'h40000000;  // 2.0
    #10;
    $display("1.0 * 2.0 = %h (Expected: 40000000)", result);
    $display("");

    // -1.0 * 2.0 (Expected result: -2.0)
    a_in = 32'hBF800000;  // -1.0 
    b_in = 32'h40000000;  // 2.0 
    #10;
    $display("-1.0 * 2.0 = %h (Expected: C0000000)", result);
    $display("");

    // 0 * 2.0 (Expected result: 0)
    a_in = 32'h00000000;  
    b_in = 32'h40000000;  
    #10;
    $display("0 * 2.0 = %h (Expected: 00000000)", result);
    $display("");

    // 1.0 * 0 (Expected result: 0)
    a_in = 32'h3F800000;  // 1.0 
    b_in = 32'h00000000;  // 0.0 
    #10;
    $display("1.0 * 0 = %h (Expected: 00000000)", result);
    $display("");

    // max positive values to check overflow
    a_in = 32'h7F7FFFFF;  
    b_in = 32'h7F7FFFFF; 
    #10;
    $display("Max * Max = %h (Overflow)", result);
	$display("");
    
    // min positive subnormal number
    a_in = 32'h00000001;  
    b_in = 32'h00000001;  
    #10;
    $display("Subnormal * Subnormal = %h", result);
    $display("");

    // largest numbers
    a_in = 32'h7EFFFFF; 
    b_in = 32'h7EFFFFF;  
    #10;
    $display("Large * Large = %h", result);
	$display("");
    
    $finish;
  end

  initial begin
    $monitor("a_in = %h, b_in = %h, result = %h", a_in, b_in, result);
  end

endmodule
