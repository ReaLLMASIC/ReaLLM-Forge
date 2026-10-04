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

module fexp (
    input [`BIT_W-1:0] x,        
    output reg [`BIT_W-1:0] result 
);    

    parameter [`BIT_W-1:0] ONE = 32'h3f800000; // 1.0
    parameter [`BIT_W-1:0] C2 = 32'h3f000000;  // 0.5
    parameter [`BIT_W-1:0] C3 = 32'h3d2aaaab;  // 1/6
    parameter [`BIT_W-1:0] C4 = 32'h3ab60b61;  // 1/24
    parameter [`BIT_W-1:0] C5 = 32'h38800000;  // 1/120

    wire [`BIT_W-1:0] term1, term2, term3, term4, term5, result_temp;
    wire [`BIT_W-1:0] x2, x3, x4, x5;

    // Compute each denominator
    fmul fmul_inst1 (.a_in(x), .b_in(x), .result(x2)); // x^2
    fmul fmul_inst2 (.a_in(x2), .b_in(x), .result(x3)); // x^3
    fmul fmul_inst3 (.a_in(x3), .b_in(x), .result(x4)); // x^4
    fmul fmul_inst4 (.a_in(x4), .b_in(x), .result(x5)); // x^5

    // Compute each term of the series
    assign term1 = x;                 // x
    fmul fmul_term2 (.a_in(x2), .b_in(C2), .result(term2)); // x^2 / 2!
    fmul fmul_term3 (.a_in(x3), .b_in(C3), .result(term3)); // x^3 / 3!
    fmul fmul_term4 (.a_in(x4), .b_in(C4), .result(term4)); // x^4 / 4!
    fmul fmul_term5 (.a_in(x5), .b_in(C5), .result(term5)); // x^5 / 5!

    // Sum all  01 terms, including the constant 1
    fadd_sub fadd_sub1(.a_operand(ONE), .b_operand(term1), .sub(0), .result(result_temp));
    fadd_sub fadd_sub2(.a_operand(result_temp), .b_operand(term2), .sub(0), .result(result_temp));
    fadd_sub fadd_sub3(.a_operand(result_temp), .b_operand(term3), .sub(0), .result(result_temp));
    fadd_sub fadd_sub4(.a_operand(result_temp), .b_operand(term4), .sub(0), .result(result_temp));
    fadd_sub fadd_sub5(.a_operand(result_temp), .b_operand(term5), .sub(0), .result(result));

endmodule

