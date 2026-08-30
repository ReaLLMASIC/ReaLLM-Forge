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

// =============================================================================
// Accumulation Top Module

module core_acc #(
    parameter   IDATA_BIT = 32, // Set a Higher Bitwidth for Accumulation
    parameter   ODATA_BIT = 32,
    parameter   CDATA_BIT = 8   // Config Bitwidth
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Global Config Signals
    input       [CDATA_BIT-1:0] cfg_acc_num,

    // Data Signals
    input       [IDATA_BIT-1:0] idata,
    input                       idata_valid,
    output      [ODATA_BIT-1:0] odata,
    output                      odata_valid
);

    // Accumulation Counter
    wire    pre_finish;

    core_acc_ctrl   #(.CDATA_BIT(CDATA_BIT)) acc_counter_inst (
        .clk                (clk),
        .rst                (rst),
        .cfg_acc_num        (cfg_acc_num),
        .psum_valid         (idata_valid),
        .psum_finish        (pre_finish)
    );

    // Accumulation Logic
    core_acc_mac    #(.IDATA_BIT(IDATA_BIT), .ODATA_BIT(ODATA_BIT)) acc_mac_inst (
        .clk                (clk),
        .rst                (rst),
        .pre_finish         (pre_finish),

        .idata              (idata),
        .idata_valid        (idata_valid),
        .odata              (odata),
        .odata_valid        (odata_valid)
    );

endmodule

// =============================================================================
// FSM for Accumulation Counter

module core_acc_ctrl #(
    parameter   CDATA_BIT = 8
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Config Signals
    input       [CDATA_BIT-1:0] cfg_acc_num,

    // Control Signals
    input                       psum_valid,
    output  reg                 psum_finish
);

    parameter   PSUM_IDLE   = 2'b01,
                PSUM_UPDATE = 2'b10;
    reg [1:0]   psum_state;

    reg     [CDATA_BIT-1:0] psum_cnt;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            psum_state  <= 2'b0;
            psum_cnt    <= 'd0;
            psum_finish <= 1'b0;
        end
        else begin
            case (psum_state)
                PSUM_IDLE: begin
                    psum_cnt    <= 'd0;
                    psum_finish <= 1'b0;
                    if (psum_valid) begin
                        psum_state <= PSUM_UPDATE;
                    end
                end
                PSUM_UPDATE: begin
                    if (psum_cnt == cfg_acc_num) begin
                        if (psum_valid) begin
                            psum_state  <= PSUM_UPDATE;
                            psum_cnt    <= 'd0;
                            psum_finish <= 1'b1;
                        end
                        else begin
                            psum_state  <= PSUM_IDLE;
                            psum_cnt    <= 'd0;
                            psum_finish <= 1'b1;
                        end
                    end
                    else begin
                        psum_cnt    <= psum_cnt + 1'b1;
                        psum_finish <= 1'b0;
                    end
                end
                default: begin
                    psum_state <= PSUM_IDLE;
                end
            endcase
        end
    end

endmodule

// =============================================================================
// Computing Logic in Accumulation

module core_acc_mac #(
    parameter   IDATA_BIT = 32, 
    parameter   ODATA_BIT = 32  // Note: ODATA_BIT >= IDATA_BIT
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Control Signals
    input                       pre_finish,

    // Data Signals
    input       [IDATA_BIT-1:0] idata,
    input                       idata_valid,
    output  reg [ODATA_BIT-1:0] odata,
    output  reg                 odata_valid
);

    // Input Gating
    reg signed  [IDATA_BIT-1:0] idata_reg;
    reg                         idata_valid_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            idata_reg <= 'd0;
        end
        else if (idata_valid) begin
            idata_reg <= idata;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            idata_valid_reg <= 1'b0;
        end
        else begin
            idata_valid_reg <= idata_valid;
        end
    end

    // Accumulation
    reg signed  [ODATA_BIT-1:0] acc_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_reg <= 'd0;
        end
        else if (idata_valid_reg) begin
            if (pre_finish) begin
                acc_reg <= 'd0;
            end
            else begin
                acc_reg <= idata_reg + acc_reg;
            end
        end
    end

    // Output and Valid
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata <= 'd0;
        end
        else if (pre_finish) begin
            odata <= idata_reg + acc_reg;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata_valid <= 1'b0;
        end
        else begin
            odata_valid <= pre_finish;
        end
    end

endmodule