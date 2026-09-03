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

module counter_tb;
  parameter MAX_COUNT = 31;
  localparam BIT_WIDTH = $clog2(MAX_COUNT + 1);

  reg clk;
  reg rstn;
  reg inc;

  wire overflow;
  wire [BIT_WIDTH-1:0] out;

  counter #(
    .MAX_COUNT(MAX_COUNT)
  ) count (
    .clk(clk),
    .rstn(rstn),
    .inc(inc),
    .overflow(overflow),
    .out(out)
  );

  initial begin
    clk = 0;
    forever #5 clk = ~clk; // 10ns period
  end

  initial begin
    rstn = 0;
    inc = 0;

    // Apply reset
    #10;
    rstn = 1;
    #10;
    rstn = 0;
    #10;
    rstn = 1;

    // increment
    #10;
    inc = 1;
    #400;

    // stay the same
    inc = 0;
    #20;

    $finish;
  end

  // Monitor outputs
  initial begin
    $monitor("time %t, clk = %b, rstn = %b, inc = %b, out = %d, overflow = %b",
              $time, clk, rstn, inc, out, overflow);
  end

endmodule
