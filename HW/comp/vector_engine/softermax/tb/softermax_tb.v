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


module softermax_tb;

  // Parameters
  parameter DATA_SIZE = 16; // Adjust according to your design
  parameter LARGE_SIZE = 32; // Adjust according to your design
  parameter ROW_WIDTH = 8; // Adjust according to your design

  // Signals
  reg signed rst_n;
  reg signed clk;
  reg signed input_valid;
  reg signed [DATA_SIZE-1:0] input_vector;
  reg signed norm_valid;
  reg signed final_out_valid;
  reg signed [LARGE_SIZE:0] prob_buffer;
  reg [$clog2(`ROW_WIDTH)-1:0] read_addr;
  // Instantiate the module
  softermax u_sf_norm (
    .rst_n(rst_n),
    .clk(clk),
    .input_valid(input_valid),
    .read_addr(read_addr),
    .input_vector(input_vector),
    .norm_valid(norm_valid),
    .final_out_valid(final_out_valid),
    .prob_buffer_out(prob_buffer)
  );

  // Clock Generation
  always begin
    #5 clk = ~clk;
  end

  // Reset Generation
  initial begin
    rst_n = 0;
    #10 rst_n = 1;
  end

  // Test Scenario
  initial begin
    // Apply test vectors here
    clk = 1;
    // Example:
    // Initialize input_vector and input_valid
    for(int j=0; j<10; j++) begin
      rst_n = 1;
      for(int i=0; i<20; i++) begin
          input_vector = 16'b0;
          input_vector[DATA_SIZE-4:0] = $random;
          input_valid = 1;

          // Wait for normalization to complete (assuming you know the timing)
          #10;

          // Check the results
          if (norm_valid) begin
          $display("Normalization completed successfully!");
          $display("Final Output Valid: %b", final_out_valid);
          // Display prob_buffer values
          for (int i = 0; i < ROW_WIDTH; i = i + 1) begin
              $display("prob_buffer[%0d]: %h", i, prob_buffer[i]);
          end
          end else begin
          $display("Normalization failed!");
          end
          input_valid = 0;
      end
      rst_n = 0;
    #20;
    end

    // End simulation
    $finish;
  end

  // Add more test scenarios as needed

endmodule
