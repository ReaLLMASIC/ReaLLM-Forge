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

module counter
#(
    parameter MAX_COUNT=31,
    localparam BIT_WIDTH=$clog2(MAX_COUNT+1)
)
(
    input clk,
    input rstn,
    input inc,
    //input new_inst,
    output overflow,
    output logic [BIT_WIDTH-1:0] out
);
// logic inc_d;
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        out <= 0;
//      inc_d <= 0;
    end
    else begin
//      inc_d <= inc;
	    //if ((inc&&overflow) || new_inst) out<=0;
        if ((inc && overflow) ) out <= 0;
	    else if (inc) out <= out+1;
	    else out <= out;
    end
end

// assign overflow=(out==MAX_COUNT) & inc_d;
assign overflow=(out==MAX_COUNT) & inc;

endmodule
