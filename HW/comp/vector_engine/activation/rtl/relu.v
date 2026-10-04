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

//ReLU function: f(x) = max(0, x)
//ReLU paper: https://arxiv.org/pdf/1803.08375

`ifndef __RELU_V_
`define __RELU_V_

module relu #(
    parameter   I_EXP   = 8,
    parameter   I_MNT   = 23,
    parameter   I_DATA  = I_EXP + I_MNT + 1
)(
    input       [I_DATA-1:0]    idata,
    output reg  [I_DATA-1:0]    odata
);

    always @(*) begin
        odata = (idata[I_DATA-1])? I_DATA'h0: idata;
    end

endmodule

`endif

