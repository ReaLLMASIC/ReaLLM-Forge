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

module tb_silu;

    parameter I_EXP   = 8;
    parameter I_MNT   = 23;
    parameter I_DATA  = I_EXP + I_MNT + 1;

    reg  [I_DATA-1:0] idata;
    wire [I_DATA-1:0] odata;

    silu #(
        .I_EXP(I_EXP),
        .I_MNT(I_MNT),
        .I_DATA(I_DATA)
    ) dut (
        .idata(idata),
        .odata(odata)
    );

    initial begin
        // idata > 4
        idata = 32'h41000000; 
        #10;
        $display("idata > 4");
        $display("expected: 8.0");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // 0 < idata < 4
        idata = 32'h40800000; 
        #10;
        $display("0 < idata < 4");
      	$display("expected: 4.0");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // -4 < idata < 0 (-2.0)
        idata = 32'hc0000000; 
        #10;
        $display("-4 < idata < 0 (-2.0)");
      $display("expected: -0.25(32'hBE000000)");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // idata < -4
        idata = 32'hc1000000; 
        #10;
        $display("idata < -4");
        $display("expected: 0.0");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // idata is zero
        idata = 32'h00000000; 
        #10;
        $display("idata is zero");
        $display("expected: 0.0");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // idata is positive infinity
        idata = 32'h7f800000; 
        #10;
        $display("idata is positive infinity");
        $display("expected: +Infinity");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // idata is negative infinity
        idata = 32'hff800000; 
        #10;
        $display("idata is negative infinity");
        $display("expected: 0.0");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        // idata is NaN
        idata = 32'h7fc00000; 
        #10;
        $display("idata is NaN");
        $display("expected: NaN");
        $display("idata: %h, odata: %h", idata, odata);
        $display("");

        $finish;
    end

endmodule
