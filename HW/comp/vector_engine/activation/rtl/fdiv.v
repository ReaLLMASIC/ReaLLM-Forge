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

module fdiv (
    input [31:0] a,  
    input [31:0] b,  
    output reg [31:0] result 
);

    wire sign_a = a[31];
    wire sign_b = b[31];
    wire [7:0] exp_a = a[30:23];
    wire [7:0] exp_b = b[30:23];
    wire [23:0] mant_a = {1'b1, a[22:0]}; 
    wire [23:0] mant_b = {1'b1, b[22:0]}; 

    wire sign_result = sign_a ^ sign_b;

    // Check for zero inputs
    wire a_is_zero = (exp_a == 8'h00) && (a[22:0] == 23'h000000);
    wire b_is_zero = (exp_b == 8'h00) && (b[22:0] == 23'h000000);

    // Exponent calculation
    wire [8:0] exp_temp = exp_a - exp_b + 127; // Subtract exponents and add bias

    // Mantissa division
    reg [47:0] mant_result;
    always @(*) begin
        if (b_is_zero) begin
            // x/0: result should be infinity or NaN
            mant_result = 48'hFFFFFF;
        end else begin
            mant_result = (mant_a << 24) / mant_b; // Align for precision and divide
        end
    end

    // Normalize result
    reg [7:0] exp_result;
    reg [22:0] mantissa_result;
    always @(*) begin
        if (a_is_zero || b_is_zero) begin
            // If either input is zero, result is zero
            exp_result = 8'h00;
            mantissa_result = 23'h000000;
        end else if (mant_result[47]) begin
            // If the result's leading bit is 1, shift right
            exp_result = exp_temp + 1;
            mantissa_result = mant_result[46:24];
        end else begin
            // No shift needed
            exp_result = exp_temp;
            mantissa_result = mant_result[45:23];
        end
    end

    // Handle overflow, underflow, and special cases
    wire overflow = (exp_result >= 255);
    wire underflow = (exp_result <= 0);
    always @(*) begin
        if (overflow) begin
            // Overflow results in infinity
            result = {sign_result, 8'hFF, 23'h000000};
        end else if (underflow) begin
            // Underflow results in zero
            result = {sign_result, 8'h00, 23'h000000};
        end else if (b_is_zero) begin
            // Division by zero results in infinity or NaN
            result = {sign_result, 8'hFF, mantissa_result};
        end else begin
            // Normal case
            result = {sign_result, exp_result, mantissa_result};
        end
    end
endmodule
