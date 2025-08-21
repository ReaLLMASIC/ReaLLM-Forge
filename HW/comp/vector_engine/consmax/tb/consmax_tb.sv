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

module consmax_tb ();

parameter   IDATA_BIT = 8;  // Input Data in INT
parameter   ODATA_BIT = 8;  // Output Data in INT
parameter   CDATA_BIT = 8;  // Global Config Data
parameter   EXP_BIT = 8;    // Exponent
parameter   MAT_BIT = 7;    // Mantissa
parameter   LUT_DATA  = EXP_BIT + MAT_BIT + 1;  // LUT Data Width (in FP)
parameter   LUT_ADDR  = IDATA_BIT >> 1;         // LUT Address Width
parameter   LUT_DEPTH = 2 ** LUT_ADDR;           // LUT Depth for INT2FP

logic                       clk;
logic                       rstn;
logic       [CDATA_BIT-1:0] cfg_consmax_shift;
logic       [LUT_ADDR:0]    cons_lut_waddr;
logic                       cons_lut_wen;
logic       [LUT_DATA-1:0]  cons_lut_wdata;
logic       [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0][`ARR_IDATA_BIT-1:0] cons_idata;
logic       [`ARR_HNUM-1:0]                                                        cons_idata_valid;
logic       [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0][`ARR_IDATA_BIT-1:0] cons_odata;
logic       [`ARR_HNUM-1:0][`ARR_GBUS_DATA/`ARR_IDATA_BIT-1:0]                     cons_odata_valid;

always_comb begin
    cons_idata = 'b0;
    cons_idata_valid = 'b0;
    for(int i=0;i<`ARR_HNUM;i++) begin
        if(ctrl_cons_valid[i]) begin
            for(int j=0;j<`ARR_VNUM;j++) begin
                if(gbus_rvalid[i][j]) begin
                    cons_idata[i]=gbus_rdata[i][j];
                    cons_idata_valid[i] =  1'b1;
                end
            end
        end
    end
end

consmax #(
    .IDATA_BIT(`ARR_IDATA_BIT),
    .ODATA_BIT(`ARR_IDATA_BIT),
    .CDATA_BIT(`ARR_CDATA_BIT),
    .EXP_BIT(8),
    .MAT_BIT(7)
) consmax_top(
    .clk(clk),
    .rstn(rstn),
    .cfg_consmax_shift(cfg_consmax_shift),
    .lut_waddr(cons_lut_waddr),//spi1
    .lut_wen(cons_lut_wen),//spi1
    .lut_wdata(cons_lut_wdata),//spi1
    .idata(cons_idata),
    .idata_valid(cons_idata_valid),
    .odata(cons_odata),
    .odata_valid(cons_odata_valid)
);
endmodule
