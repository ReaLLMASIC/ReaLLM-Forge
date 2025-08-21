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

`define LN_NUM 32
`define INV_LN_NUM 1023410176 // value of 1/LN_NUM in fp8
module RMS_layer_norm #(
    parameter sig_width = 23,
    parameter exp_width = 8,
    parameter ieee_compliance = 0,
    parameter en_ubr_flag = 0,
    parameter MAC_NUM = 8,
    parameter isign = 1,
    parameter INT_WIDTH = 8,
    parameter FP_WIDTH = 32,
    parameter DATA_DEPTH = 8,
    parameter isize = INT_WIDTH,
    parameter OUT_INT_WIDTH = 8,
    parameter RF_ADDR = 7,
    parameter RF_BITS = 32
) ( 
    input clk, rst_n,
    input [INT_WIDTH-1:0] input_data_array [DATA_DEPTH-1:0],
    //input [DATA_DEPTH*INT_WIDTH-1:0] input_data,
    //input  [FP_WIDTH-1:0] gamma ,
    input  valid_in,

    //inputs of cache                    
    input cache_wen,
    input [RF_ADDR-1:0] cache_waddr,
    input [RF_BITS-1:0] cache_wdata,
    output wire valid_out,
    //output reg [DATA_DEPTH*OUT_INT_WIDTH-1:0] ln_out
    output reg [INT_WIDTH-1:0] ln_out_array [DATA_DEPTH-1:0]
    //output fifo_full, fifo_empty //for debug
);  
    
    wire [DATA_DEPTH*INT_WIDTH-1:0] input_data;
    reg [DATA_DEPTH*OUT_INT_WIDTH-1:0] ln_out ;
    genvar i;
    generate
        for(i=0; i<DATA_DEPTH; i=i+1) begin
            assign input_data[(i+1)*INT_WIDTH-1:i*INT_WIDTH] = input_data_array[i];
            assign ln_out_array[i] = ln_out[(i+1)*OUT_INT_WIDTH-1:i*OUT_INT_WIDTH];
        end
    endgenerate
    
    reg [RF_ADDR-1:0] cache_raddr;
    reg [RF_BITS-1:0] cache_rdata;
    reg cache_ren;
    layernorm_lut kv_cache_inst(
        .ickwp0(clk),
        .iwenp0(cache_wen),
        .iawp0(cache_waddr),
        .idinp0(cache_wdata),
        .ickrp0(clk),
        .irenp0(cache_ren),
        .iarp0(cache_raddr),
        .iclkbyp(1'b1),
        .imce(1'b0),
        .irmce(2'b0),
        .ifuse(1'b1),
        .iwmce(4'b0),
        .odoutp0(cache_rdata)
    );

    genvar j;
    //adder tree used to add all 8 outputs from core together
    reg adder_tree_out_valid;
    reg [FP_WIDTH-1:0] adder_tree_out;
    reg [FP_WIDTH*DATA_DEPTH-1:0] sqr_output_reg;
    adder_tree1  #(.MAC_NUM(MAC_NUM), 
                    .IDATA_BIT(FP_WIDTH), 
                    .ODATA_BIT(FP_WIDTH),
                    .sig_width(sig_width),
                    .exp_width(exp_width),
                    .ieee_compliance(ieee_compliance)
                    )adder_tree_inst (
        .clk                            (clk),
        .rstn                           (rst_n),
        .idata                          (sqr_output_reg),
        .idata_valid                    (valid_in),
        .odata                          (adder_tree_out),
        .odata_valid                    (adder_tree_out_valid)
    );
    //counter to count how may data_in has been added to sum reg
    reg [$clog2(`LN_NUM)-1:0] counter1;

    ////////////////////////////////////
    /// transfer input data into FP8 ///
    ////////////////////////////////////
    reg [8*DATA_DEPTH-1:0]status_flt1;
    reg [FP_WIDTH*DATA_DEPTH-1:0] buffer_data_in_reg;
    generate
        for(j=0; j<DATA_DEPTH; j=j+1) begin: int_to_float
        DW_fp_i2flt #(sig_width, exp_width, isize, isign)
            DW_fp_i2flt ( .a(input_data[(j+1)*INT_WIDTH-1:j*INT_WIDTH]), .rnd(3'b000), .z(buffer_data_in_reg[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .status(status_flt1[(j+1)*8-1:j*8]) );
        end
    endgenerate

    //////////////////////////////////////////
    /// calculate the square of input data ///
    //////////////////////////////////////////
    reg [8*DATA_DEPTH-1:0] status_sqr;
    reg [FP_WIDTH*DATA_DEPTH-1:0] sqr_in_reg;
    wire [FP_WIDTH*DATA_DEPTH-1:0] sqr_output;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            sqr_in_reg <= 0;
            sqr_output_reg <= 0;
        end
        else begin
            sqr_in_reg <= buffer_data_in_reg;
            sqr_output_reg <= sqr_output;
        end
    end 
    generate
        for(j=0; j<DATA_DEPTH; j=j+1) begin
            DW_fp_mult #(sig_width, exp_width, ieee_compliance, en_ubr_flag)
                mult_before_addertree ( .a(buffer_data_in_reg[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .b(buffer_data_in_reg[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .rnd(3'b000), .z(sqr_output[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .status(status_sqr[(j+1)*8-1:j*8]) );
        end
    endgenerate

    /////////////////////////////////////////////
    /// transfer output of addertree into FP8 ///
    /////////////////////////////////////////////
    
    reg [FP_WIDTH-1:0] adder_tree_out_reg;
    reg adder_tree_out_valid_reg, adder_tree_out_valid_reg_0;
    //reg [FP_WIDTH-1:0] sum_input_data;
    //reg [7:0] status_flt2;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            adder_tree_out_reg <= 0;
            adder_tree_out_valid_reg <= 0;
            adder_tree_out_valid_reg_0 <= 0;
        end
        else begin
            adder_tree_out_reg <= adder_tree_out;
            adder_tree_out_valid_reg_0 <= adder_tree_out_valid;
            adder_tree_out_valid_reg <= adder_tree_out_valid_reg_0;
        end
    end

    //////////////////////////////////////////
    /// buffer used to hold all input data ///
    //////////////////////////////////////////
    //wire full, empty;
    reg wr_en,rd_en;
    reg [FP_WIDTH*DATA_DEPTH-1:0] buffer_data_in;
    wire [FP_WIDTH*DATA_DEPTH-1:0] buffer_data_out;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            wr_en <= 1'b0;
            rd_en <= 1'b0;
            buffer_data_in <= 8'b0;
        end
        else begin
            if(adder_tree_out_valid_reg) begin
                buffer_data_in <= buffer_data_in_reg;
                wr_en <= 1;//need more consideration
                if(counter1 == `LN_NUM-DATA_DEPTH) begin
                    rd_en <= 1;
                end
            end
            else begin
                //buffer_data_in <= 0;
                wr_en <= 0;
            end
        end
    end

    buffer #(
        .DATA_WIDTH(FP_WIDTH),
        .FIFO_DEPTH(DATA_DEPTH*DATA_DEPTH),
        .INPUT_WIDTH(DATA_DEPTH),  // New parameter for input width
        .OUTPUT_WIDTH(DATA_DEPTH)  // New parameter for output width
    )b1 (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .rd_en(rd_en),
        .data_in(buffer_data_in),
        .data_out(buffer_data_out),
        .full(fifo_full),
        .empty(fifo_empty)
    );

    ////////////////////////////////
    /// control logic of adder 1 ///
    ////////////////////////////////
    
    //adder 1 counts the summation of data_in;
    reg [FP_WIDTH-1:0] inst_a1, inst_b1;
    wire [FP_WIDTH-1:0] z_inst1;
    reg [FP_WIDTH-1:0]sum1;
    wire [7:0] status_add_1;
    reg start;//start signal to start the computation
    always @ (posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            sum1 <= 0;
            inst_a1 <= 0;
            inst_b1 <= 0;
            counter1 <= 0;
            start <= 0;
            cache_raddr <= 0;
        end
        else begin
            if(!adder_tree_out_valid_reg) begin
                sum1 <= 0;
                inst_a1 <= 0;
                inst_b1 <= 0;
                counter1 <= 0;
                start <= 0;
                cache_raddr <= 0;
                cache_ren <= 0;
            end
            else begin
                if(adder_tree_out_valid_reg && (counter1 == `LN_NUM-DATA_DEPTH || start == 0)) begin
                    counter1 <= 0;
                    inst_a1 <= 0;
                    //sum1 <= 0;
                    start <= 1;
                    cache_raddr <= cache_raddr+1;
                    cache_ren <= 1;
                    inst_b1 <= adder_tree_out_reg;
                end
                else begin
                    counter1 <= counter1 + DATA_DEPTH;
                    inst_a1 <= z_inst1;    
                    inst_b1 <= adder_tree_out_reg;
                end
                sum1 <= z_inst1; 
            end
        end
    end

    DW_fp_add #(sig_width, exp_width, ieee_compliance)
        add1 ( .a(inst_a1), .b(inst_b1), .rnd(3'b000), .z(z_inst1), .status(status_add_1) );
    
    /////////////////////////////////////
    /// control logic of multiplier 2 ///
    /////////////////////////////////////
    
    reg [FP_WIDTH-1:0] inst_a3, inst_b3;
    wire [FP_WIDTH-1:0] z_inst3;
    reg [FP_WIDTH-1:0] mean1;
    reg [$clog2(`LN_NUM):0] counter1_reg;
    wire [7:0] status_mult_2;
    //this multiplier will get the mean of sum1(sum of data_in)
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            inst_a3 <= 0;
            inst_b3 <= 0;
            mean1 <= 0;
            counter1_reg <= 0;
        end
        else begin
            if(counter1_reg == `LN_NUM-DATA_DEPTH) begin
                inst_a3 <= sum1;
                inst_b3 <= `INV_LN_NUM;//need transfer to 1/LN_NUM in floating point
            end
            counter1_reg <= counter1;
            mean1 <= z_inst3;
        end
    end
    DW_fp_mult #(sig_width, exp_width, ieee_compliance, en_ubr_flag)
        mult2 ( .a(inst_a3), .b(inst_b3), .rnd(3'b000), .z(z_inst3), .status(status_mult_2) );  
    
    ////////////////////////////////
    /// control logic of invsqrt ///
    ////////////////////////////////
    wire [FP_WIDTH-1:0] z_inst7;
    reg [FP_WIDTH-1:0] sqr_div;
    wire [7:0] status_invsqrt_1;
    //calculate the invsqrt of var
    always @ (posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            sqr_div <= 0;
        end
        else begin
            sqr_div <= z_inst7;
        end
    end

    DW_fp_invsqrt #(sig_width, exp_width, ieee_compliance)
        invsqrt1 ( .a(mean1), .rnd(3'b000), .z(z_inst7), .status(status_invsqrt_1) );

    /////////////////////////////////////
    /// control logic of multiplier 5 ///
    /////////////////////////////////////
    wire [FP_WIDTH*DATA_DEPTH-1:0] z_inst9;
    reg [FP_WIDTH*DATA_DEPTH-1:0] temp_out;
    wire [8*DATA_DEPTH-1:0] status_mult_5;
    reg [FP_WIDTH*DATA_DEPTH-1:0] data_out_reg, data_out_reg0;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            temp_out <= 0;
            data_out_reg <= 0;
            data_out_reg0 <= 0;
        end
        else begin
            temp_out <= z_inst9;
            data_out_reg <= data_out_reg0;
            data_out_reg0 <= buffer_data_out;
        end
    end
    generate
        for(j=0; j<DATA_DEPTH; j=j+1) begin
            DW_fp_mult #(sig_width, exp_width, ieee_compliance, en_ubr_flag)
                mult5 ( .a(data_out_reg[(1+j)*FP_WIDTH-1:j*FP_WIDTH]), .b(sqr_div), .rnd(3'b000), .z(z_inst9[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .status(status_mult_5[(j+1)*8-1:j*8]) );
        end
    endgenerate

    /////////////////////////////////////
    /// control logic of multiplier 6 ///
    /////////////////////////////////////
    wire [FP_WIDTH*DATA_DEPTH-1:0] z_inst10;
    reg [FP_WIDTH*DATA_DEPTH-1:0] FP_ln_out;
    wire [8*DATA_DEPTH-1:0] status_mult_6;
    reg [FP_WIDTH-1:0] gamma, gamma_reg0, gamma_reg1;
    //this multiplier will generate the gamma*x
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            FP_ln_out <= 0;
            gamma <= 0;
            gamma_reg0 <= 0;
            gamma_reg1 <= 0;
        end
        else begin
            FP_ln_out <= z_inst10;
            gamma_reg0 <= cache_rdata;
            gamma_reg1 <= gamma_reg0;
            gamma <= gamma_reg1;
        end
    end
    generate
        for(j=0; j<DATA_DEPTH; j=j+1) begin
            DW_fp_mult #(sig_width, exp_width, ieee_compliance, en_ubr_flag)
                mult6 ( .a(temp_out[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .b(gamma), .rnd(3'b000), .z(z_inst10[(j+1)*FP_WIDTH-1:j*FP_WIDTH]), .status(status_mult_6[(j+1)*8-1:j*8]) );
        end
    endgenerate

    //////////////////////////////////
    /// control logic of FP to INT ///
    //////////////////////////////////
    reg [8*DATA_DEPTH-1:0] status_flt22;
    localparam isize3 = OUT_INT_WIDTH;
    reg [DATA_DEPTH*OUT_INT_WIDTH-1:0] z_inst12;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            ln_out <= 0;
        end
        else begin
            ln_out <= z_inst12;
        end
    end

    generate
        for(j=0; j<DATA_DEPTH; j=j+1) begin : float_to_int
            DW_fp_flt2i #(sig_width, exp_width, isize3, ieee_compliance) 
                flt2i1 (.a(FP_ln_out[(j+1)*FP_WIDTH-1:j*FP_WIDTH]),.rnd(3'b000),.z(z_inst12[(j+1)*OUT_INT_WIDTH-1:j*OUT_INT_WIDTH]),.status(status_flt22[(j+1)*8-1:j*8]) );
        end
    endgenerate

    reg [12:0] shift_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg <= 13'b0;
        end else begin
            shift_reg <= {shift_reg[11:0], adder_tree_out_valid};
        end
    end

    // Assign the output to the leftmost bit of the shift register
    assign valid_out = shift_reg[12];

    
endmodule
