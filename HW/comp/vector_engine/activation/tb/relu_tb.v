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

module tb_relu;

    parameter I_EXP   = 8;
    parameter I_MNT   = 23;
    parameter I_DATA  = I_EXP + I_MNT + 1;

    reg [I_DATA-1:0] idata;
    wire [I_DATA-1:0] odata;

    relu #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) dut (
        .idata(idata),
        .odata(odata)
    );

    initial begin
        idata = 0;

        // Test positive value
        idata = 32'b0_01111111_00000000000000000000000; // +1.0 in IEEE 754
        #10;
        $display("Positive value");
        $display("expected: 0 01111111 00000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test negative value
        idata = 32'b1_01111111_00000000000000000000000; // -1.0 in IEEE 754
        #10;
        $display("Negative value");
        $display("expected: 00000000000000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test zero value
        idata = 32'b0_00000000_00000000000000000000000; // +0.0 in IEEE 754
        #10;
        $display("Zero value");
        $display("expected: 0 00000000 00000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test negative zero value
        idata = 32'b1_00000000_00000000000000000000000; // -0.0 in IEEE 754
        #10;
        $display("Negative zero value");
        $display("expected: 00000000000000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test positive infinity
        idata = 32'b0_11111111_00000000000000000000000; // +inf in IEEE 754
        #10;
        $display("Positive infinity");
        $display("expected: 0 11111111 00000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test negative infinity
        idata = 32'b1_11111111_00000000000000000000000; // -inf in IEEE 754
        #10;
        $display("Negative infinity");
        $display("expected: 00000000000000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test positive NaN
        idata = 32'b0_11111111_10000000000000000000000; // NaN in IEEE 754
        #10;
        $display("Positive NaN");
        $display("expected: 0 11111111 10000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        // Test negative NaN
        idata = 32'b1_11111111_10000000000000000000000; // -NaN in IEEE 754
        #10;
        $display("Negative NaN");
        $display("expected: 00000000000000000000000000000000");
        $display("%b -> %b", idata, odata);
        $display("");

        $finish;
    end

endmodule
