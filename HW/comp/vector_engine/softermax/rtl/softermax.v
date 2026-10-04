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

`define DATA_SIZE 16
`define FRAC 4
`define LARGE_SIZE 16
`define N 64
`define M 64
`define ROW_WIDTH 8
`define USE_PIPELINE 1
module softermax (
    input rst_n,
    input clk,
    input input_valid,
    input signed [`DATA_SIZE-1:0] input_vector,
    input [$clog2(`ROW_WIDTH)-1:0]read_addr,
    //valid flag for normalization
    output reg norm_valid, //should set input_valid to 0, before finish normalization
    //1 means normalizaiton finished
    output reg final_out_valid,
    //buffer to hold the output value from normalization
    output reg [`LARGE_SIZE:0] prob_buffer_out
);  
    reg [`LARGE_SIZE:0] prob_buffer [`ROW_WIDTH-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prob_buffer_out<=0;
        end
        else
            prob_buffer_out<=prob_buffer[read_addr];
    end


    reg signed [`DATA_SIZE-1:0] global_max;
    //sf_counter for sf unit
    reg [$clog2(`ROW_WIDTH):0] sf_counter;
    //norm_counter for norm unit
    reg [$clog2(`ROW_WIDTH):0] norm_counter;
    //flag shown sf_unit is valid
    reg output_valid;
    //input and output of sf_unit
    reg signed [`DATA_SIZE-1:0] pre_max;
    reg signed [`LARGE_SIZE-1:0] pre_denominator;
    reg signed [`DATA_SIZE-1:0] current_max;
    reg signed [`LARGE_SIZE-1:0] current_denominator;
    reg signed [`LARGE_SIZE:0] uSoftmax;
    //max and uSoftmax buffer
    reg signed [`DATA_SIZE-1:0] local_max [`ROW_WIDTH-1:0];
    reg signed [`LARGE_SIZE:0] local_uSoftmax [`ROW_WIDTH-1:0];
    //output from normalization
    reg signed [`LARGE_SIZE-1:0] prob;
    

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_max <= 0;
            sf_counter <= 0;
            norm_valid <= 0;
            pre_max <= 0;
            pre_denominator <= 0;
        end
        else begin
            if(output_valid) begin
                if(sf_counter[$clog2(`ROW_WIDTH)] == 1) begin
                    sf_counter <= 0;
                    norm_valid <= 1;
                end
                else begin
                    sf_counter <= sf_counter + 1;
                end
                if(current_max > global_max) begin
                    global_max <=  current_max;
                end
                //update buffer
                local_max[sf_counter] <= current_max;
                local_uSoftmax[sf_counter] <= uSoftmax;

                //update data for sf_unit
                pre_max <= current_max;
                pre_denominator <= current_denominator;
            end
        end
    end

    sf_unit sf_unit1(
        .input_valid(input_valid),
        .pre_max(pre_max),
        .input_vector(input_vector),
        .pre_denominator(pre_denominator),
        .output_valid(output_valid),
        .current_max(current_max),
        .uSoftmax(uSoftmax),
        .clk(clk),
        .rst_n(rst_n),
        .current_denominator(current_denominator)
    );

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            norm_counter <= 0;
            final_out_valid <= 0;
        end
        else begin
            if(norm_valid) begin
                 if(norm_counter[$clog2(`ROW_WIDTH)] == 1) begin
                    norm_counter <= 0;
                    final_out_valid <= 1;
                end
                else begin
                    norm_counter <= norm_counter + 1;
                end
                //prob_buffer[norm_counter] <= prob;
            end
        end
    end

    always @(posedge clk) begin
        if(norm_valid)
            prob_buffer[norm_counter] <= prob;
    end

    normalization norm_unit(
        .clk(clk),
        .rst_n(rst_n),
        .uSoftmax(local_uSoftmax[norm_counter]),
        .local_max(local_max[norm_counter]),
        .global_max(global_max),
        .current_denominator(current_denominator),
        .prob(prob)
    );
endmodule

module sf_unit(
    input clk,
    input rst_n,
    input input_valid,
    input signed [`DATA_SIZE-1:0] pre_max,
    input signed [`DATA_SIZE-1:0] input_vector,
    input signed [`LARGE_SIZE-1:0] pre_denominator,
    output reg output_valid,
    output reg signed [`DATA_SIZE-1:0] current_max,
    output reg signed [`LARGE_SIZE:0] uSoftmax,
    output reg signed [`LARGE_SIZE-1:0] current_denominator
);

    reg [`DATA_SIZE-1:0] current_max_reg;
    reg [`DATA_SIZE-1:0] input_vector_reg;
    reg input_valid_d;
    reg [`DATA_SIZE-1:0] pre_max_temp;
    reg [`LARGE_SIZE-1:0] pre_denominator_temp;
    reg [`LARGE_SIZE-1:0] current_denominator_temp;
    reg [`LARGE_SIZE:0] uSoftmax_temp;
    reg [`DATA_SIZE-1:0] current_max_temp;
    reg output_valid_temp;

    IntMax IntMax0 (
        .pre_max(pre_max),
        .input_vector(input_vector),
        .current_max(current_max_temp)
    );
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_max_reg<=0;
            input_vector_reg<=0;
            input_valid_d<=0;
            pre_max_temp<=0;
            pre_denominator_temp<=0;
        end
        else begin
            current_max_reg<=current_max_temp;
            input_vector_reg<=input_vector;
            input_valid_d<=input_valid;
            pre_max_temp<=pre_max;
            pre_denominator_temp<=pre_denominator;
        end
    end
    
    Pow2 Pow_unit (
        .clk(clk),
        .rst_n(rst_n),
        .current_max(current_max_reg),
        .input_vector(input_vector_reg),
        .uSoftmax(uSoftmax_temp)
    );  
    
    //debug logic
    reg signed [`DATA_SIZE-1:0] temp5;
    reg signed [`DATA_SIZE-1:0] temp6;
    reg signed [`DATA_SIZE-1:0] temp7;

    always @(*) begin
        if(input_valid_d) begin
            temp5 = current_max>pre_max_temp?(current_max-pre_max_temp):0;
            temp6 = pre_denominator_temp >>> temp5;
            temp7 = temp6 + uSoftmax_temp;
            current_denominator_temp = temp7;
            output_valid_temp = 1;
        end
        else begin
            current_denominator_temp = 0;
            output_valid_temp = 0;
        end
    end

    //output stage
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_valid<=0;
            current_max<=0;
            uSoftmax<=0;
            current_denominator<=0;
        end
        else begin
            output_valid<=output_valid_temp;
            current_max<=current_max_reg;
            uSoftmax<=uSoftmax_temp;
            current_denominator<=current_denominator_temp;
        end
    end

    
endmodule

module normalization(
    input clk,
    input rst_n,
    input signed [`LARGE_SIZE:0] uSoftmax,
    input signed [`DATA_SIZE-1:0] local_max,
    input signed [`DATA_SIZE-1:0] global_max,
    input signed [`LARGE_SIZE-1:0] current_denominator,
    output signed [`LARGE_SIZE-1:0] prob
);
    wire signed [`LARGE_SIZE-1:0] divisor;
    reg signed [`LARGE_SIZE-1:0] divisor_reg;
    // reg signed [`LARGE_SIZE-1:0] FP_1;
    //reg signed [`DATA_SIZE-1:`FRAC] int_part;
    //always @(*) begin
    //    FP_1 = 0;
    //    FP_1[`FRAC] = 1;
    //end
    // assign FP_1[`FRAC] = 1;
    // assign FP_1[`LARGE_SIZE-1:`FRAC+1] = 0;
    // assign FP_1[`FRAC-1:0] = 0;
    //reg signed [`LARGE_SIZE-1:0]
    
    assign divisor = uSoftmax  >>> (global_max - local_max);
    generate
        if(`USE_PIPELINE == 1) begin
            always @(posedge clk or negedge rst_n) begin
                if(!rst_n) begin
                    divisor_reg <= 0;
                end
                else begin
                    divisor_reg <= divisor;
                end
            end
            assign prob = divisor_reg*current_denominator;
        end
        else begin
            assign prob = divisor*current_denominator;
        end
    endgenerate
    
endmodule

//perform 2^(xj-localMax)
//To do: modify output data size
module Pow2 (
    input clk,
    input rst_n,
    input signed [`DATA_SIZE-1:0] current_max,
    input signed [`DATA_SIZE-1:0] input_vector,
    output wire [`LARGE_SIZE:0] uSoftmax //UnnormedSoftmax
);  

    wire signed [`DATA_SIZE-`FRAC-1:0] integer_part;
    wire signed [`FRAC-1:0] fraction_part;
    wire signed [`DATA_SIZE-1:0] pow2_frac;
    wire signed [`LARGE_SIZE-1:0] FP_1;
    assign integer_part = input_vector[`DATA_SIZE-1:`FRAC];
    assign fraction_part = input_vector[`FRAC-1:0];

    LUT LUT0 (
        .index(input_vector[`FRAC-1:0]),
        .out(pow2_frac)
    );
    wire signed[`DATA_SIZE-1:0] temp1;
    wire signed[`DATA_SIZE-1:0] temp2;
    wire signed[`DATA_SIZE-1:0] temp3;
    reg signed [`DATA_SIZE-1:0]temp1_reg;
    reg signed [`DATA_SIZE-1:0]temp2_reg;
    generate
        if(`USE_PIPELINE == 1) begin
            always @(posedge clk or negedge rst_n) begin
                if(!rst_n) begin
                    temp1_reg <= 0;
                    temp2_reg <= 0;
                end
                else begin
                    temp1_reg <= temp1;
                    temp2_reg <= temp2;
                end
            end

            assign temp1 = (integer_part > current_max)?integer_part - current_max:current_max-integer_part;
            assign temp2 = (integer_part > current_max)?1<<<temp1_reg:1>>>temp1_reg;
            assign temp3 = temp2_reg*pow2_frac;
            assign uSoftmax = temp3;
        end
        else begin
            assign temp1 = (integer_part > current_max)?integer_part - current_max:current_max-integer_part;
            assign temp2 = (integer_part > current_max)?1<<<temp1:1>>>temp1;
            assign temp3 = temp2*pow2_frac;
            assign uSoftmax = temp3;
        end
    endgenerate
endmodule


module LUT(
    input signed [`FRAC-1:0] index,
    output reg signed [`DATA_SIZE-1:0] out
);
    always @(*) begin
        //[2^(n/16)] and then round to integer
        out = 8'b0;
        case(index)
            4'b0000 : out = 8'b00010000;
            4'b0001 : out = 8'b00010001;
            4'b0010 : out = 8'b00010001;
            4'b0011 : out = 8'b00010010;
            4'b0100 : out = 8'b00010011;
            4'b0101 : out = 8'b00010100;
            4'b0110 : out = 8'b00010101;
            4'b0111 : out = 8'b00010110;
            4'b1000 : out = 8'b00010111;
            4'b1001 : out = 8'b00011000;
            4'b1010 : out = 8'b00011001;
            4'b1011 : out = 8'b00011010;
            4'b1100 : out = 8'b00011011;
            4'b1101 : out = 8'b00011100;
            4'b1110 : out = 8'b00011101;
            4'b1111 : out = 8'b00011111;          
        endcase
    end
endmodule

module fixed_point_division(
    input  signed [`LARGE_SIZE-1:0] a,  // 8-bit input a, 4-bit integer and 3-bit fractional part
    input  signed [`LARGE_SIZE-1:0] b,  // 8-bit input b, 4-bit integer and 3-bit fractional part
    output signed [`LARGE_SIZE-1:0] result  // 8-bit result, 4-bit integer and 3-bit fractional part
);
    // Extend the bits to avoid overflow during division
    // Adjust the scaling by shifting the numerator (a) to left by the number of fractional bits
    wire signed [`LARGE_SIZE+`LARGE_SIZE-1:0] extended_a;
    wire signed [`LARGE_SIZE+`LARGE_SIZE-1:0] extended_b;
    wire signed [`LARGE_SIZE+`LARGE_SIZE-1:0] extended_result;

    assign extended_a = a[`LARGE_SIZE-1] ? (~a <<<`FRAC):(a <<< `FRAC);
    assign extended_b = b[`LARGE_SIZE-1] ? (~b):b;

    // Perform the division
    divider divider0 (
        .dividend(extended_a),
        .divisor(extended_b),
        .quotient(extended_result)
    );

    // Scale back the result by right shifting 
    // and truncating to fit into the original bit width

    wire sign_flag;
    assign sign_flag = (a[`LARGE_SIZE-1]==b[`LARGE_SIZE-1])?0:1;
    assign result = sign_flag ? (~extended_result) : extended_result;
    //assign result = extended_result;
endmodule

module divider (
    input signed [`N-1:0] dividend,
    input signed [`M-1:0] divisor,
    output reg signed [`N-1:0] quotient
);

   // Width of the divisor
reg signed [`M-1:0] remainder;
integer i;

always @(dividend or divisor) begin
    quotient = 0;
    remainder = 0;
    for (i = `N-1; i >= 0; i = i - 1) begin
        remainder = remainder << 1;
        remainder[0] = dividend[i];
        if (remainder >= divisor) begin
            remainder = remainder - divisor;
            quotient[i] = 1;
        end
    end
end

endmodule


module IntMax (
    input signed [`DATA_SIZE-1:0] pre_max,
    input signed [`DATA_SIZE-1:0] input_vector,
    output signed [`DATA_SIZE-1:0] current_max
);
    reg signed [`DATA_SIZE-1:0] int_input;

    //perform ceiling function, only keelp integer part and plus 1
    always @(*) begin : ceiling_function
        if(input_vector[`FRAC-1:0] != 0) begin
            int_input = input_vector[`DATA_SIZE-1:`FRAC] + 1;
        end
        else begin
            int_input = input_vector[`DATA_SIZE-1:`FRAC];
        end
    end

    assign current_max = (int_input > pre_max) ? int_input : pre_max;
endmodule
