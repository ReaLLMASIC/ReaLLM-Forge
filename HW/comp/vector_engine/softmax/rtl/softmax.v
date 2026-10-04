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

`define DATA_BIT 16 //input and output data width
`define CDATA_BIT 8

`define SOFTMAX_NUM 8 //softmax total count.
`define SOFTMAX_ADDR $clog2(`SOFTMAX_NUM)

`define I_EXP 8
`define I_MAT 7
`define LUT_ADDR 9
`define LUT_DEPTH 2 ** `LUT_ADDR
`define LUT_DATA `I_EXP + `I_MAT + 1

module softmax
(
    // Global Signals
    input                       clk,
    input                       rst,

    // Control Signals
    input       [`CDATA_BIT-1:0] cfg_consmax_shift,

    // LUT Interface
    input       [`LUT_ADDR-1:0] lut_waddr,
    input                       lut_wen,
    input       [`LUT_DATA-1:0]  lut_wdata,

    // Data Signals
    input       [`DATA_BIT-1:0] idata,
    input                       idata_valid,
    output  reg [`DATA_BIT-1:0] odata,
    output  reg                 odata_valid
);  
    wire [18-1:0] fadd_in_a;
    wire [18-1:0] fadd_in_b;
    wire [18-1:0] fadd_out;
	//input
    reg     [`DATA_BIT-1:0] idata_reg [`SOFTMAX_NUM-1:0];
    reg                     idata_valid_reg;
    reg     [`SOFTMAX_ADDR-1:0] pointer;
    wire    full;
	reg     [`DATA_BIT-1:0] comp_in;

    assign full=(pointer==`SOFTMAX_NUM-1);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pointer<= 1'd0;
        end
        else if (idata_valid && !full) begin
            pointer<=pointer+1;
        end
        else if (full) begin
            pointer<=0;
        end
    end

	integer i;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for(i=0;i<`SOFTMAX_NUM;i=i+1)
                idata_reg[i] <= 'd0;
			// comp_in<=0;
        end
        else if (idata_valid) begin
            idata_reg[pointer] <= idata;
			// comp_in<=idata;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            idata_valid_reg <= 1'b0;
        end
        else begin
            idata_valid_reg <= idata_valid;
        end
    end

    //compare for max_buffer then update the max buffer
    reg [`DATA_BIT-1:0] max_buffer_in;
    reg [`DATA_BIT-1:0] max_buffer;
    assign max_buffer_sign = max_buffer[`DATA_BIT-1];
    assign max_buffer_exp = max_buffer[`DATA_BIT-2 -: `I_EXP];
    assign max_buffer_m = max_buffer[`I_MAT-1:0];

    assign idata_sign = idata[`DATA_BIT-1];
    assign idata_exp = idata[`DATA_BIT-2 -: `I_EXP];
    assign idata_m = idata[`I_MAT-1:0];

    assign max_buffer_in= (idata_sign<max_buffer_sign) ? idata : (idata_sign==max_buffer_sign)&&(idata_exp>max_buffer_exp) ? idata : (idata_sign==max_buffer_sign)&&(idata_exp==max_buffer_exp)&&(idata_m > max_buffer_m) ? idata : max_buffer;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            max_buffer <= 1'd0;
        end
        else if(full)
            max_buffer <= 1'd0;
        else if(idata_valid_reg) begin
            max_buffer <= max_buffer_in;
        end
    end

    //max buffer needs to propagate
    // genvar i;
    // reg [`DATA_BIT-1:0] max_reg [`SOFTMAX_NUM-1:0];
    // generate
    //     for (i = 0; i < REG_NUM; i = i + 1) begin
    //         if(i==0) begin
    //             always @(posedge clk or posedge rst) begin
    //                 if (rst) 
    //                     max_reg[0]<='0;
    //                 else if(full)
    //                     max_reg[0]<=max_buffer;
    //             end
    //         end
    //         else begin
    //             always @(posedge clk or posedge rst) begin
    //                 if (rst) 
    //                     max_reg[i]<='0;
    //                 else
    //                     max_reg[i]<=max_reg[i-1];
    //             end
    //         end
    //     end
    // endgenerate
    reg [`DATA_BIT-1:0] max_reg;
    always @(posedge clk or posedge rst) begin
        if (rst) 
            max_reg<='0;
        else if(full)
            max_reg<=max_buffer;
    end


    //getting stored input and subtract with max

    reg     [`SOFTMAX_ADDR-1:0] rd_pointer;
    wire     rd_full;
	reg 	buffer_valid;
	reg [`DATA_BIT-1:0] lut_addr_fp;
	reg lut_addr_fp_valid;

    assign rd_full=(rd_pointer==`SOFTMAX_NUM-1);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_pointer<= 1'd0;
        end
        else if (full) begin
            rd_pointer<=rd_pointer+1;
        end
        else if(rd_full) begin
            rd_pointer<=0;
        end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            buffer_valid <= 1'b0;
        end
        else
            buffer_valid <=full;
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            lut_addr_fp<= 1'd0;
        end
        else if (buffer_valid) begin
            lut_addr_fp<=fadd_out;
        end

    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            lut_addr_fp_valid <= 1'b0;
        end
        else begin
            lut_addr_fp_valid <= buffer_valid;
        end
    end

    //fp input to fixed point
    reg [`LUT_ADDR-1:0] lut_addr_comb;
    reg [`LUT_ADDR-1:0] lut_addr;
    reg lut_addr_valid;

    fp2int fp2int_inst (
        .cfg_shift                  (cfg_consmax_shift),
        .idata                      (lut_addr_fp),
        .odata                      (lut_addr_comb)
    ); 

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            lut_addr <= 'd0;
        end
        else if (lut_addr_fp_valid)begin
            lut_addr <= lut_addr_comb;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            lut_addr_valid <= 1'b0;
        end
        else begin
            lut_addr_valid <= lut_addr_fp_valid;
        end
    end

    //lut based exponential
    wire    [`LUT_ADDR-1:0] lut_addr_w;
    wire                    lut_ren;
    wire    [`LUT_DATA-1:0]  lut_rdata;
    reg     exp_valid;

    mem_sp lut_inst (
        .clk                    (clk),
        .addr                   (lut_addr_w),
        .wen                    (lut_wen),
        .wdata                  (lut_wdata),
        .ren                    (lut_ren),
        .rdata                  (lut_rdata)
    ); 
    
    assign  lut_addr_w = lut_wen ? lut_waddr : lut_addr;
    assign  lut_ren  = lut_wen ? 1'b0      : lut_addr_valid;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_valid <= 1'b0;
        end
        else begin
            exp_valid <= lut_addr_valid;
        end
    end

	//store exponent
	reg [`LUT_DATA-1:0] exp_reg [`SOFTMAX_NUM-1:0];
	reg  [`SOFTMAX_ADDR-1:0] exp_pointer;
	wire exp_full;
	reg exp_reg_valid; // all exponent have been stored.

	assign exp_full=(exp_pointer==`SOFTMAX_NUM-1);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_pointer<= 1'd0;
        end
        else if (exp_valid && !exp_full) begin
            exp_pointer<=exp_pointer+1;
        end
        else if(exp_full) begin
            exp_pointer<=0;
        end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            for(i=0;i<`SOFTMAX_NUM;i=i+1)
                exp_reg[i] <= 'd0;
        end
        else if (exp_valid) begin
            exp_reg[exp_pointer] <= lut_rdata;
        end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_reg_valid <= 1'b0;
        end
        else begin
            exp_reg_valid <= exp_full;
        end
    end

	//read exponent and accumulate
    reg     [`SOFTMAX_ADDR-1:0] exp_rd_pointer;
    wire     exp_rd_full;
	reg 	exp_read_valid;
	reg [`LUT_DATA-1:0] exp_acc; //accumulated partial sum
	reg sum_valid; //exp sum valid

    assign exp_rd_full=(rd_pointer==`SOFTMAX_NUM-1);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_rd_pointer<= 1'd0;
        end
        else if (exp_read_valid && !exp_rd_full) begin
            exp_rd_pointer<=exp_rd_pointer+1;
        end
        else if(rd_full) begin
            exp_rd_pointer<=0;
        end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_read_valid <= 1'b0;
        end
        else if(exp_reg_valid) begin
            exp_read_valid <= 1'b1;
        end
		else if(exp_rd_full) begin
			exp_read_valid <=1'b0;
		end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            exp_acc<= 1'd0;
        end
        else if (exp_read_valid) begin
            exp_acc<=fadd_out;
        end
	end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            sum_valid <= 1'b0;
        end
        else begin
            sum_valid <= exp_rd_full;
        end
    end

	//get reciprocal for exp_acc
	wire [18-1:0] finv_in;
	wire [18-1:0] finv_out;

	assign finv_in=exp_acc;
	fpinv finv(.f1(finv_in),.fout(finv_out));
	
	reg [`LUT_DATA-1:0] denominator;
	reg deno_valid;


	always @(posedge clk or posedge rst) begin
        if (rst) begin
            denominator<= 1'd0;
        end
        else if(sum_valid) begin
            denominator<=finv_out; //need to change, denominator will be covered by next pipeline input
        end
    end
	always @(posedge clk or posedge rst) begin
        if (rst) begin
            deno_valid <= 1'b0;
        end
        else begin
            deno_valid <= sum_valid;
        end
    end

	//divide each exp stored
	reg     [`SOFTMAX_ADDR-1:0] out_pointer;
    wire     out_full;
	reg 	out_valid;

    assign out_full=(out_pointer==`SOFTMAX_NUM-1);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_pointer<= 1'd0;
        end
        else if (out_valid && !out_full) begin
            out_pointer<=out_pointer+1;
        end
        else if(rd_full) begin
            out_pointer<=0;
        end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_valid <= 1'b0;
        end
        else if(deno_valid) begin
            out_valid <= 1'b1;
        end
		else if(out_full) begin
			out_valid <=1'b0;
		end
    end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata<= 1'd0;
        end
        else if (odata_valid) begin
            odata<=fadd_out;
        end
	end

	always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata_valid <= 1'b0;
        end
        else begin
            odata_valid <= out_valid;
        end
    end

	//fmul for output
	wire [18-1:0] fpmult_in1;
	wire [18-1:0] fpmult_in2;
	wire [18-1:0] fpmult_out; 
	assign fpmult_in1=exp_reg[out_pointer];
	assign fpmult_in2=denominator;
	fpmult fmul (.f1(fpmult_in1),.f2(fpmult_in2),.fout(fpmult_out));
	
    //fadd for comp, subtract, accumulate

    fpadd fadd(fadd_out,fadd_in_a,fadd_in_b);
    assign fadd_in_a= buffer_valid ? idata_reg[rd_pointer] : exp_read_valid ? exp_reg[exp_rd_pointer] : exp_reg[exp_rd_pointer];
    assign fadd_in_b= buffer_valid ? {!(max_reg[`DATA_BIT-1]),max_reg[`DATA_BIT-2:0]} : exp_read_valid ? exp_acc : exp_acc;  

endmodule

// =============================================================================
// Floating-Point Multiplier

//////////////////////////////////////////////////////////
// floating point multiply 
// -- sign bit -- 8-bit exponent -- 9-bit mantissa
// Similar to fp_mult from altera
// NO denorms, no flags, no NAN, no infinity, no rounding!
//////////////////////////////////////////////////////////
// f1 = {s1, e1, m1), f2 = {s2, e2, m2)
// If either is zero (zero MSB of mantissa) then output is zero
// If e1+e2<129 the result is zero (underflow)
///////////////////////////////////////////////////////////	
module fpmult (fout, f1, f2);

	input [17:0] f1, f2 ;
	output [17:0] fout ;
	
	wire [17:0] fout ;
	reg sout ;
	reg [8:0] mout ;
	reg [8:0] eout ; // 9-bits for overflow
	
	wire s1, s2;
	wire [8:0] m1, m2 ;
	wire [8:0] e1, e2, sum_e1_e2 ; // extend to 9 bits to avoid overflow
	wire [17:0] mult_out ;	// raw multiplier output
	
	// parse f1
	assign s1 = f1[17]; 	// sign
	assign e1 = {1'b0, f1[16:9]};	// exponent
	assign m1 = f1[8:0] ;	// mantissa
	// parse f2
	assign s2 = f2[17];
	assign e2 = {1'b0, f2[16:9]};
	assign m2 = f2[8:0] ;
	
	// first step in mult is to add extended exponents
	assign sum_e1_e2 = e1 + e2 ;
	
	// build output
	// raw integer multiply
	// unsigned_mult mm(mult_out, m1, m2);
	assign mult_out=m1*m2;
	 
	// assemble output bits
	assign fout = {sout, eout[7:0], mout} ;
	
	always @(*)
	begin
		// if either is denormed or exponents are too small
		if ((m1[8]==1'd0) || (m2[8]==1'd0) || (sum_e1_e2 < 9'h82)) 
		begin 
			mout = 0;
			eout = 0;
			sout = 0; // output sign
		end
		else // both inputs are nonzero and no exponent underflow
		begin
			sout = s1 ^ s2 ; // output sign
			if (mult_out[17]==1)
			begin // MSB of product==1 implies normalized -- result >=0.5
				eout = sum_e1_e2 - 9'h80;
				mout = mult_out[17:9] ;
			end
			else // MSB of product==0 implies result <0.5, so shift ome left
			begin
				eout = sum_e1_e2 - 9'h81;
				mout = mult_out[16:8] ;
			end	
		end // nonzero mult logic
	end // always @(*)
	
endmodule

/////////////////////////////////////////////////////////////////////////////
// floating point Add 
// -- sign bit -- 8-bit exponent -- 9-bit mantissa
// NO denorms, no flags, no NAN, no infinity, no rounding!
/////////////////////////////////////////////////////////////////////////////
// f1 = {s1, e1, m1), f2 = {s2, e2, m2)
// If either input is zero (zero MSB of mantissa) then output is the remaining input.
// If either input is <(other input)/2**9 then output is the remaining input.
// Sign of the output is the sign of the greater magnitude input
// Add the two inputs if their signs are the same. 
// Subtract the two inputs (bigger-smaller) if their signs are different
//////////////////////////////////////////////////////////////////////////////	
module fpadd (fout, f1, f2);

	input [17:0] f1, f2 ;
	output [17:0] fout ;
	
	wire  [17:0] fout ;
	wire sout ;
	reg [8:0] mout ;
	reg [7:0] eout ;
	reg [9:0] shift_small, denorm_mout ; //9th bit is overflow bit
	
	wire s1, s2 ; // input signs
	reg  sb, ss ; // signs of bigger and smaller
	wire [8:0] m1, m2 ; // input mantissas
	reg  [8:0] mb, ms ; // mantissas of bigger and smaller
	wire [7:0] e1, e2 ; // input exp
	wire [7:0] ediff ;  // exponent difference
	reg  [7:0] eb, es ; // exp of bigger and smaller
	reg  [7:0] num_zeros ; // high order zeros in the difference calc
	
	// parse f1
	assign s1 = f1[17]; 	// sign
	assign e1 = f1[16:9];	// exponent
	assign m1 = f1[8:0] ;	// mantissa
	// parse f2
	assign s2 = f2[17];
	assign e2 = f2[16:9];
	assign m2 = f2[8:0] ;
	
	// find biggest magnitude
	always @(*)
	begin
		if (e1>e2) // f1 is bigger
		begin
			sb = s1 ; // the bigger number (absolute value)
			eb = e1 ;
			mb = m1 ;
			ss = s2 ; // the smaller number
			es = e2 ;
			ms = m2 ;
		end
		else if (e2>e1) //f2 is bigger
		begin
			sb = s2 ; // the bigger number (absolute value)
			eb = e2 ;
			mb = m2 ;
			ss = s1 ; // the smaller number
			es = e1 ;
			ms = m1 ;
		end
		else // e1==e2, so need to look at mantissa to determine bigger/smaller
		begin
			if (m1>m2) // f1 is bigger
			begin
				sb = s1 ; // the bigger number (absolute value)
				eb = e1 ;
				mb = m1 ;
				ss = s2 ; // the smaller number
				es = e2 ;
				ms = m2 ;
			end
			else // f2 is bigger or same size
			begin
				sb = s2 ; // the bigger number (absolute value)
				eb = e2 ;
				mb = m2 ;
				ss = s1 ; // the smaller number
				es = e1 ;
				ms = m1 ;
			end
		end
	end //found the bigger number
	
	// sign of output is the sign of the bigger (magnitude) input
	assign sout = sb ;
	// form the output
	assign fout = {sout, eout, mout} ;	
	
	// do the actual add:
	// -- equalize exponents
	// -- add/sub
	// -- normalize
	assign ediff = eb - es ; // the actual difference in exponents
	always @(*)
	begin
		if ((ms[8]==0) && (mb[8]==0))  // both inputs are zero
		begin
			mout = 9'b0 ;
			eout = 8'b0 ; 
		end
		else if ((ms[8]==0) || ediff>8)  // smaller is too small to matter
		begin
			mout = mb ;
			eout = eb ;
		end
		else  // shift/add/normalize
		begin
			// now shift but save the low order bits by extending the registers
			// need a high order bit for 1.0<sum<2.0
			shift_small = {1'b0, ms} >> ediff ;
			// same signs means add -- different means subtract
			if (sb==ss) //do the add
			begin
				denorm_mout = {1'b0, mb} + shift_small ;
				// normalize --
				// when adding result has to be 0.5<sum<2.0
				if (denorm_mout[9]==1) // sum bigger than 1
				begin
					mout = denorm_mout[9:1] ; // take the top bits (shift-right)
					eout = eb + 1 ; // compensate for the shift-right
				end
				else //0.5<sum<1.0
				begin
					mout = denorm_mout[8:0] ; // drop the top bit (no-shift-right)
					eout = eb ; // 
				end
			end // end add logic
			else // otherwise sb!=ss, so subtract
			begin
				denorm_mout = {1'b0, mb} - shift_small ;
				// the denorm_mout is always smaller then the bigger input
				// (and never an overflow, so bit 9 is always zero)
				// and can be as low as zero! Thus up to 8 shifts may be necessary
				// to normalize denorm_mout
				if (denorm_mout[8:0]==9'd0)
				begin
					mout = 9'b0 ;
					eout = 8'b0 ;
				end
				else
				begin
					// detect leading zeros
					casex (denorm_mout[8:0])
						9'b1xxxxxxxx: num_zeros = 8'd0 ;
						9'b01xxxxxxx: num_zeros = 8'd1 ;
						9'b001xxxxxx: num_zeros = 8'd2 ;
						9'b0001xxxxx: num_zeros = 8'd3 ;
						9'b00001xxxx: num_zeros = 8'd4 ;
						9'b000001xxx: num_zeros = 8'd5 ;
						9'b0000001xx: num_zeros = 8'd6 ;
						9'b00000001x: num_zeros = 8'd7 ;
						9'b000000001: num_zeros = 8'd8 ;
						default:       num_zeros = 8'd9 ;
					endcase	
					// shift out leading zeros
					// and adjust exponent
					eout = eb - num_zeros ;
					mout = denorm_mout[8:0] << num_zeros ;
				end
			end // end subtract logic
			// format the output
			//fout = {sout, eout, mout} ;	
		end // end shift/add(sub)/normailize/
	end // always @(*) to compute sum/diff	
endmodule

//////////////////////////////////////////////////////////
// floating point reciprocal  (invert)
// -- sign bit -- 8-bit exponent -- 9-bit mantissa
// Similar to fp_mult from altera
// NO denorms, no flags, no NAN, no infinity, no rounding!
//////////////////////////////////////////////////////////
// f1 = {s1, e1, m1)
// If f1 is zero, set output to max number (about 1e38)
///////////////////////////////////////////////////////////	
module fpinv (fout, f1);

	input [17:0] f1 ;
	output [17:0] fout ;
	
	wire [17:0] fout ;
	reg sout ;
	reg [8:0] mout ;
	reg [8:0] eout ; // 9-bits for overflow
	
	wire s1;
	wire [8:0] m1 ;
	wire [8:0] e1 ; // extend to 9 bits to avoid overflow
	wire [17:0] inv_out ;	// 
	
	// parse f1
	assign s1 = f1[17]; 	// sign
	assign e1 = {1'b0, f1[16:9]};	// exponent
	assign m1 = f1[8:0] ;	// mantissa
	
	// assemble output bits from 'always @' below
	assign fout = {sout, eout[7:0], mout} ;
	
	// newton iteration: linear approx + 2 steps
	// x0 = T1 - 2*input (input 0.5<=input<=1.0
	// x1 = x0*(2-input*x0)
	// x2 = x1*(2-input*x1)
	// from http://en.wikipedia.org/wiki/Division_%28digital%29
	wire [17:0] x0, x1, x2, reduced_input, reduced_input_x_2 ;
	wire [17:0] input_x_x0, input_x_x1 ;
	wire [17:0] two_minus_input_x_x0, two_minus_input_x_x1 ;
	
	parameter T1 = 18'h10575 ; // T1=2.9142
	parameter two = {1'b0, 8'h82, 9'h100} ;
	
	// form (T1-2*input)
	// BUT limit input range on 0.5 to 1.0 (just the mantissa)
	// THEN mult by two by setting exp to 8'h81
	// AND make it negative by setting the sign bit
	assign reduced_input = {1'b1, 8'h80, m1} ;
	assign reduced_input_x_2 = {1'b1, 8'h81, m1} ;
	fpadd init_newton(x0, reduced_input_x_2, T1) ;
	
	// form x1 = x0*(2-input*x0)
	fpmult newton11(input_x_x0, reduced_input, x0) ;
	fpadd newton12(two_minus_input_x_x0, two, input_x_x0);
	fpmult newton13(x1, x0, two_minus_input_x_x0) ;
	
	// form x2 = x1*(2-input*x1)
	fpmult newton21(input_x_x1, reduced_input, x1) ;
	fpadd newton22(two_minus_input_x_x1, two, input_x_x1);
	fpmult newton23(x2, x1, two_minus_input_x_x1) ;
	
	// select between zero and nonzero input
	always @(*)
	begin
		// if input is zero
		if (m1[8]==1'd0) 
		begin 
			// make the biggest possible output
			mout = 9'h100 ; 
			eout = 9'h0ff ;
			sout = 0; // output sign
		end
		
		else // input is nonzero 
		begin 
			eout = (m1==9'b100000000)? 9'h102 - e1 : 9'h101 - e1 ; // h81+(h81-e1)
			sout = s1; // output sign
			mout = x2[8:0] ; // the newton result		
		end // input is nonzero
	end
endmodule

/////////////////////////////////////////////////////////////////////////////
// floating point to integer 
// output: 10-bit signed integer 
// input: -- sign bit -- 8-bit exponent -- 9-bit mantissa 
// and scale factor (-100 to +100) powers of 2
// fp_out = {sign, exp, mantissa}
// NO denorms, no flags, no NAN, no infinity, no rounding!
/////////////////////////////////////////////////////////////////////////////
// =============================================================================
module fp2int 
(
    // Control Signals
    input       [`CDATA_BIT-1:0] cfg_shift,

    // Data Signals
    input       [`DATA_BIT-1:0] idata,
    output      [9-1:0] odata
);

    // Extrach Sign and Mantissa Field from FP
    reg                 idata_sig;
    reg     [`I_MAT:0] idata_mat;

    always @(*) begin
        idata_sig = idata[`DATA_BIT-1];
        idata_mat = {1'b1, idata[`I_MAT-1:0]};
    end

    // Shift and Round Mantissa to Integer
    reg     [`I_MAT:0] mat_shift;
    reg     [`I_MAT:0] mat_round;

    always @(*) begin
        mat_shift = idata_mat >> cfg_shift;
    end

    always @(*) begin
        mat_round = mat_shift[`I_MAT:1] + mat_shift[0];
    end

    // Convert to 2's Complementary Integer
    //assign odata = {idata_sig, idata_sig ? (~mat_round[`I_MAT-:`DATA_BIT] + 1'b1) : mat_round[`I_MAT-:`DATA_BIT]};
	assign odata = {idata_sig, mat_round};//for power est

endmodule

module mem_sp 
(
    // Global Signals
    input                       clk,

    // Data Signals
    input       [`LUT_ADDR-1:0]  addr,
    input                       wen,
    input       [`LUT_DATA-1:0]  wdata,
    input                       ren,
    output  reg [`LUT_DATA-1:0]  rdata
);

    // 1. RAM/Memory initialization
    reg [`LUT_DATA-1:0]  mem [0:`LUT_DEPTH-1];

    // 2. Write channel
    always @(posedge clk) begin
        if (wen) begin
            mem[addr] <= wdata;
        end
    end

    // 3. Read channel
    always @(posedge clk) begin
        if (ren) begin
            rdata <= mem[addr];
        end
    end

endmodule

