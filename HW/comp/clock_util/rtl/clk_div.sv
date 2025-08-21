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

module clk_div(
    input clk,
    input rst_n,
    output div2048_clk
);
reg clk_div2;
reg clk_div4;
reg clk_div8;
reg clk_div16;
reg clk_div32;
reg clk_div64;
reg clk_div128;
reg clk_div256;
reg clk_div512;
reg clk_div1024;
reg clk_div2048;
assign div2048_clk = clk_div2048;

always @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        clk_div2 <= 0;
    else
        clk_div2 <= ~clk_div2;
end

always @(posedge clk_div2 or negedge rst_n) begin
    if(~rst_n)
        clk_div4 <= 0;
    else
        clk_div4 <= ~clk_div4;
end

always @(posedge clk_div4 or negedge rst_n) begin
    if(~rst_n)
        clk_div8 <= 0;
    else
        clk_div8 <= ~clk_div8;
end

always @(posedge clk_div8 or negedge rst_n) begin
    if(~rst_n)
        clk_div16 <= 0;
    else
        clk_div16 <= ~clk_div16;
end

always @(posedge clk_div16 or negedge rst_n) begin
    if(~rst_n)
        clk_div32 <= 0;
    else
        clk_div32 <= ~clk_div32;
end

always @(posedge clk_div32 or negedge rst_n) begin
    if(~rst_n)
        clk_div64 <= 0;
    else
        clk_div64 <= ~clk_div64;
end

always @(posedge clk_div64 or negedge rst_n) begin
    if(~rst_n)
        clk_div128 <= 0;
    else
        clk_div128 <= ~clk_div128;
end

always @(posedge clk_div128 or negedge rst_n) begin
    if(~rst_n)
        clk_div256 <= 0;
    else
        clk_div256 <= ~clk_div256;
end

always @(posedge clk_div256 or negedge rst_n) begin
    if(~rst_n)
        clk_div512 <= 0;
    else
        clk_div512 <= ~clk_div512;
end

always @(posedge clk_div512 or negedge rst_n) begin
    if(~rst_n)
        clk_div1024 <= 0;
    else
        clk_div1024 <= ~clk_div1024;
end

always @(posedge clk_div1024 or negedge rst_n) begin
    if(~rst_n)
        clk_div2048 <= 0;
    else
        clk_div2048 <= ~clk_div2048;
end
endmodule
