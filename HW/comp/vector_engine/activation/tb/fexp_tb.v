`timescale 1ns / 1ps

module tb_fexp;
    reg [`BIT_W-1:0] x;

    wire [`BIT_W-1:0] result;
  	wire [`BIT_W-1:0] term1, term2, term3, term4, term5, result_temp;
  	wire [`BIT_W-1:0] x2, x3, x4, x5;
  	wire [`BIT_W-1:0] result_temp_1, result_temp_2, result_temp_3, result_temp_4;

    fexp dut (
        .x(x),
      	.result(result),
        // For Debug
      	// .term1(term1), .term2(term2), .term3(term3), .term4(term4), .term5(term5), 
      	// .x2(x2), .x3(x3), .x4(x4), .x5(x5),
      	// .result_temp_1(result_temp_1), .result_temp_2(result_temp_2), .result_temp_3(result_temp_3), .result_temp_4(result_temp_4)
    );

    initial begin
        x = 0;
        #10;

        // Test Case 1: x = 0.0 (e^0 = 1.0)
        x = 32'h00000000;
        #50;
        $display("Test 1: x = 0.0, result = 0x%08X (expected: 3f800000)", (result));

        // Test Case 2: x = 1.0 (e^1 ≈ 2.718)
        x = 32'h3f800000;
        #50;
      	$display("Test 2: x = 1.0, result = %h (expected: 402df3b6(2.718))", (result));
      	// $display("  x2 = %h, x3 = %h, x4 = %h, x5 = %h", (x2), (x3), (x4), (x5));
      	// $display("  term1 = %h, term2 = %h, term3 = %h, term4 = %h, term5 = %h", term1, (term2), (term3), (term4), (term5));
      
        // Test Case 3: x = -1.0 (e^-1 ≈ 0.367)
        x = 32'hbf800000;
        #50;
      $display("Test 3: x = -1.0, result = 0x%08X (expected: 3ebbe76d(0.367))", (result));
      // $display("  x2 = %h, x3 = %h, x4 = %h, x5 = %h", (x2), (x3), (x4), (x5));
      // $display("  term1 = %h, term2 = %h, term3 = %h, term4 = %h, term5 = %h", term1, (term2), (term3), (term4), (term5));
      
        // Test Case 4: x = 0.5 (e^0.5 ≈ 1.648)
        x = 32'h3f000000;
        #50;
      $display("Test 4: x = 0.5, result = 0x%08X (expected: 3fd2f1aa(1.648))", (result));
      // $display("  x2 = %h, x3 = %h, x4 = %h, x5 = %h", (x2), (x3), (x4), (x5));
      // $display("  term1 = %h, term2 = %h, term3 = %h, term4 = %h, term5 = %h", term1, (term2), (term3), (term4), (term5));
      
        // Test Case 5: x = -0.5 (e^-0.5 ≈ 0.607)
        x = 32'hbf000000;
        #50;
      $display("Test 5: x = -0.5, result = 0x%08X (expected: 0x3F1B4445(0.607))", (result));

      
        // Test Case 6: x = 2.0 (e^2 ≈ 7.389)
        x = 32'h40000000;
        #50;
      $display("Test 6: x = 2.0, result = 0x%08X (expected: 40e88889(7.267))", (result));
      
        // Test Case 7: x = -2.0 (e^-2 ≈ 0.135)
        x = 32'hc0000000;
        #50;
      $display("Test 7: x = -2.0, result = 0x%08X (expected: 3d888880(0.0667))", (result));
      
        // Finish simulation
        $finish;
    end

endmodule
