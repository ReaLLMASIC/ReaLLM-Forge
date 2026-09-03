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

//GeLU function: GELU(x) = xP(X <= x) = x Φ(x) ≈ 0.5x(1 + tanh[(sqrt(2 / PI)(x + 0.044715 x^3)])
//GeLU paper: https://arxiv.org/pdf/1606.08415v3.pdf 

`ifndef __GELU_V_
`define __GELU_V_

module gelu #(
    parameter   I_EXP   = 8,
    parameter   I_MNT   = 23,
    parameter   I_DATA  = I_EXP + I_MNT + 1
)(
    input       [I_DATA-1:0]    idata,
    output reg  [I_DATA-1:0]    odata
);

    // this need to be changed on fp16
    localparam const_0p04 = 32'h3D37_2713;
    localparam const_2divpi_m2 = 32'h3FCC_422A;
    localparam const_1  = {1'b0, 8'd127, {I_MNT{1'b0}}};

    wire [I_DATA-1:0] x_div2;
    wire [I_DATA-1:0] x_mul_0p04;
    wire [I_DATA-1:0] x_sq;
    wire [I_DATA-1:0] x_3;
    wire [I_DATA-1:0] x_add;
    wire [I_DATA-1:0] x_tanh;
    wire [I_DATA-1:0] x_exp;
    wire [I_DATA-1:0] x_exp_sub;
    wire [I_DATA-1:0] x_exp_add;
    wire [I_DATA-1:0] x_div;
    wire [I_DATA-1:0] x_tanh_p1;

    DW_fp_mult #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_mul_0p04 (     // 0.044715x
        .a(idata),
        .b(const_0p04),
        .rnd(3'b000),
        .z(x_mul_0p04),
        .status()
    );

    DW_fp_mult #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_sq (       // x^2
        .a(idata),
        .b(idata),
        .rnd(3'b000),
        .z(x_sq),
        .status()
    );

    DW_fp_mult #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_3 (        // 0.044715x^3
        .a(x_mul_0p04),
        .b(x_sq),
        .rnd(3'b000),
        .z(x_3),
        .status()
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_add (      // x+0.044715x^3
        .a(x_3),
        .b(idata),
        .z(x_add),
        .rnd(3'b000),
        .op(1'b0),
        .status()
    );

    DW_fp_mult #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_tanh (     // sqrt(2/pi) * ()
        .a(const_2divpi_m2),
        .b(x_add),
        .rnd(3'b000),
        .z(x_tanh),
        .status()
    );

    DW_fp_exp #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fp_x_exp (        // (exp(x) - 1) / (exp(x) + 1)
        .a(x_tanh),
        .z(x_exp),
        .status()
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_exp_sub (  // exp(x)-1
        .a(x_exp),
        .b(const_1),
        .z(x_exp_sub),
        .rnd(3'b000),
        .op(1'b1),
        .status()
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_exp_add (  // exp(x)+1
        .a(x_exp),
        .b(const_1),
        .z(x_exp_add),
        .rnd(3'b000),
        .op(1'b0),
        .status()
    );

    DW_fp_div #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fp_x_div (        // tanh()
        .a(x_exp_sub),
        .b(x_exp_add),
        .rnd(3'b000),
        .z(x_div),
        .status()
    );

    DW_fp_addsub #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_x_tanh_p1 (  // 1+tanh()
        .a(x_div),
        .b(const_1),
        .z(x_tanh_p1),
        .rnd(3'b000),
        .op(1'b0),
        .status()
    );

    fdiv2 #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) fdiv2_x_div2 (    // x/2
        .if32(idata),
        .of32(x_div2)
    );

    DW_fp_mult #(
        .sig_width(I_MNT),
        .exp_width(I_EXP)
    ) fmul_out (        // final output
        .a(x_div2),
        .b(x_tanh_p1),
        .rnd(3'b000),
        .z(odata),
        .status()
    );

endmodule
`endif

