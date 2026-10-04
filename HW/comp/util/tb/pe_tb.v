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

module tb_fp_arithmetic;

    // Parameters
    localparam IDATA_BIT = 8;
    localparam ODATA_BIT = IDATA_BIT*2;
    localparam EXP_BIT = 8;
    localparam MAT_BIT = 7;
    localparam DATA_BIT = EXP_BIT + MAT_BIT + 1;

    // Inputs and Outputs for mul_int
    reg [IDATA_BIT-1:0] idataA_int, idataB_int;
    wire [ODATA_BIT-1:0] odata_int;

    // Inputs and Outputs for add_int
    reg [IDATA_BIT-1:0] idataA_add_int, idataB_add_int;
    wire [IDATA_BIT:0] odata_add_int;

    // Inputs and Outputs for mul_fp
    reg [DATA_BIT-1:0] idataA_fp_mul, idataB_fp_mul;
    wire [DATA_BIT-1:0] odata_fp_mul;

    // Inputs and Outputs for add_fp
    reg clk, rst;
    reg [DATA_BIT-1:0] idataA_fp_add, idataB_fp_add;
    wire [DATA_BIT-1:0] odata_fp_add;

    // Instantiate mul_int
    mul_int #(
        .IDATA_BIT(IDATA_BIT),
        .ODATA_BIT(ODATA_BIT)
    ) uut_mul_int (
        .idataA(idataA_int),
        .idataB(idataB_int),
        .odata(odata_int)
    );

    // Instantiate add_int
    add_int #(
        .IDATA_BIT(IDATA_BIT),
        .ODATA_BIT(IDATA_BIT+1)
    ) uut_add_int (
        .idataA(idataA_add_int),
        .idataB(idataB_add_int),
        .odata(odata_add_int)
    );

    // Instantiate mul_fp
    mul_fp #(
        .EXP_BIT(EXP_BIT),
        .MAT_BIT(MAT_BIT),
        .DATA_BIT(DATA_BIT)
    ) uut_mul_fp (
        .idataA(idataA_fp_mul),
        .idataB(idataB_fp_mul),
        .odata(odata_fp_mul)
    );

    // Instantiate add_fp
    add_fp #(
        .EXP_BIT(EXP_BIT),
        .MAT_BIT(MAT_BIT),
        .DATA_BIT(DATA_BIT),
        .ENABLE_PIPELINE(1)
    ) uut_add_fp (
        .clk(clk),
        .rst(rst),
        .idataA(idataA_fp_add),
        .idataB(idataB_fp_add),
        .odata(odata_fp_add)
    );

    // Clock generation
    always #5 clk = ~clk;

    initial begin
        // Initialize clock and reset
        clk = 0;
        rst = 1;
        #10;
        rst = 0;

        // Test cases for mul_int
        idataA_int = 8'd15; idataB_int = 8'd10;
        #10;
        $display("INT MUL: %d * %d = %d (Expected: 150)", idataA_int, idataB_int, odata_int);
      	$display("");

        idataA_int = 8'b11101100; //-20
      	idataB_int = 8'b11111011; //-5
      	#10;
        
      $display("INT MUL: -20 * -5 = %d (Expected: 100)", odata_int);
        $display("");

        // Test cases for add_int
        idataA_add_int = 8'd20; idataB_add_int = 8'd30;
        #10;
        $display("INT ADD: %d + %d = %d (Expected: 50)", idataA_add_int, idataB_add_int, odata_add_int);
      	$display("");

        idataA_add_int = 8'b11101100; //-20
      	idataB_add_int = 8'd30;
        #10;
      $display("INT ADD: -20 + %d = %d (Expected: 10)", idataB_add_int, odata_add_int);
      	$display("");

     	// Test cases for mul_fp
        idataA_fp_mul = 16'h3C00; // 1.0 
        idataB_fp_mul = 16'h4000; // 2.0 
        #10;
       $display("FP MUL: 1.0 * 2.0 = %h (Expected: 4000)", odata_fp_mul);
      	$display("");

        idataA_fp_mul = 16'h4200; // 3.0 
        idataB_fp_mul = 16'h4200; // 3.0 
        #10;
        $display("FP MUL: 3.0 * 3.0 = %h (Expected: 4880)", odata_fp_mul);
      	$display("");

        // Test cases for add_fp
        idataA_fp_add = 16'h3C00; // 1.0 
        idataB_fp_add = 16'h4000; // 2.0 
        #10;
      	$display("FP ADD: 1.0 + 2.0 = %h (Expected: 4200)", odata_fp_add);
      	$display("");

        idataA_fp_add = 16'h4200; // 3.0 
        idataB_fp_add = 16'h4200; // 3.0 
        #10;
        $display("FP ADD: 3.0 + 3.0 = %h (Expected: 4600)", odata_fp_add);
      	$display("");
      
        $finish;
    end

    initial begin
        $monitor("Time: %0d, INT MUL: %d * %d = %d, INT ADD: %d + %d = %d, FP MUL: %h * %h = %h, FP ADD: %h + %h = %h",
                 $time, idataA_int, idataB_int, odata_int, idataA_add_int, idataB_add_int, odata_add_int,
                 idataA_fp_mul, idataB_fp_mul, odata_fp_mul, idataA_fp_add, idataB_fp_add, odata_fp_add);
    end

endmodule
