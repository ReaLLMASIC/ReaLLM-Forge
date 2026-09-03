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

module fadd_sub #(
    parameter I_EXP = 8,
    parameter I_MNT = 23,
    parameter I_DATA = I_EXP + I_MNT + 1
)(
    input [I_DATA-1:0] a_operand,
    input [I_DATA-1:0] b_operand,
    input sub, // New input to indicate subtraction
    output [I_DATA-1:0] result
);

    wire [I_EXP-1:0] exp_a, exp_b;
    wire [I_MNT:0] significand_a, significand_b;
    wire [I_EXP:0] exp_diff;
    reg [I_EXP:0] exp_result;
    reg [I_MNT+1:0] significand_result;
    wire sign_a, sign_b;
    reg sign_result;
    reg [I_DATA-1:0] result_temp;
    wire a_larger;

    assign sign_a = a_operand[I_DATA-1];
    assign sign_b = b_operand[I_DATA-1] ^ sub; // Invert sign of b_operand if subtraction
    assign exp_a = a_operand[I_DATA-2:I_MNT];
    assign exp_b = b_operand[I_DATA-2:I_MNT];
    assign significand_a = (exp_a == 0) ? {1'b0, a_operand[I_MNT-1:0]} : {1'b1, a_operand[I_MNT-1:0]};
    assign significand_b = (exp_b == 0) ? {1'b0, b_operand[I_MNT-1:0]} : {1'b1, b_operand[I_MNT-1:0]};

    // Compare exponents and significands
    assign a_larger = (exp_a > exp_b) || ((exp_a == exp_b) && (significand_a >= significand_b));

    // Align significands based on exponent difference
    assign exp_diff = a_larger ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [I_MNT:0] aligned_significand_a, aligned_significand_b;
    assign aligned_significand_a = a_larger ? significand_a : (significand_a >> exp_diff);
    assign aligned_significand_b = a_larger ? (significand_b >> exp_diff) : significand_b;

    // Add or subtract significands based on signs
    always @* begin
        if (sign_a == sign_b) begin
            significand_result = aligned_significand_a + aligned_significand_b;
            sign_result = sign_a;
        end else begin
            if (aligned_significand_a >= aligned_significand_b) begin
                significand_result = aligned_significand_a - aligned_significand_b;
                sign_result = sign_a;
            end else begin
                significand_result = aligned_significand_b - aligned_significand_a;
                sign_result = sign_b;
            end
        end
    end

    // Normalize result
    reg [I_MNT+1:0] significand_norm;
    reg [I_EXP:0] exp_adjust;
    always @* begin
        significand_norm = significand_result;
        exp_adjust = 0;
        if (significand_result[I_MNT+1]) begin
            significand_norm = significand_result >> 1;
            exp_adjust = 1;
        end else begin
            while (significand_norm[I_MNT] == 0 && exp_result > 0) begin
                significand_norm = significand_norm << 1;
                exp_adjust = exp_adjust - 1;
            end
        end
    end

    // Handle zero result
    wire zero_result = (significand_result == 0);

    assign exp_result = zero_result ? 0 :
                        (a_larger ? (exp_a + exp_adjust) : (exp_b + exp_adjust));

    // Handle special cases: NaN and Infinity
    wire is_nan, is_inf;
    assign is_nan = (&exp_a && |a_operand[I_MNT-1:0]) || (&exp_b && |b_operand[I_MNT-1:0]);
    assign is_inf = (&exp_a && ~|a_operand[I_MNT-1:0]) || (&exp_b && ~|b_operand[I_MNT-1:0]);

    always @* begin
        if (zero_result) begin
            result_temp = 32'h00000000; // Explicit zero output
        end else if (is_nan) begin
            result_temp = {1'b0, {(I_EXP){1'b1}}, {(I_MNT-1){1'b0}}, 1'b1}; // NaN
        end else if (is_inf) begin
            result_temp = {sign_result, {(I_EXP){1'b1}}, {(I_MNT){1'b0}}}; // Infinity
        end else begin
            result_temp = {sign_result, exp_result[I_EXP-1:0], significand_norm[I_MNT-1:0]};
        end
    end

    // Final output
    assign result = result_temp;

endmodule
