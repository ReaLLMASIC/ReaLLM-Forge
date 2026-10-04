
// =======================================================================================
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
// =======================================================================================

module controller_testbench();

    logic                                             clk;
    logic                                             rstn;
    CFG_ARR_PACKET                                    arr_cfg;

    logic       [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-1:0]   gbus_addr;
    CTRL                                              gbus_wen;
    CTRL                                              gbus_ren;

    logic                                              vlink_enable;
    logic       [`ARR_VNUM-1:0]                       vlink_wen;

    logic       [`ARR_HNUM-1:0]                       hlink_wen;

    logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]    global_sram_waddr;
    logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]    global_sram_raddr;
    logic                                             global_sram_wen;
    logic                                             global_sram_ren;
    GSRAM_WSEL                                         global_sram_wsel;
    GSRAM_RSEL                                         global_sram_rsel;

    CMEM_ARR_PACKET                                   arr_cmem;

    CTRL                                              lbuf_empty;
    CTRL                                              lbuf_full;
    CTRL                                              lbuf_ren;

    CTRL                                              abuf_empty;
    CTRL                                              abuf_full;
    CTRL                                              abuf_ren;

    CTRL                                              gbus_rvalid;
    logic       [`ARR_HNUM-1:0]                       ctrl_cons_ovalid;

    controller #(
        .INST_REG_DEPTH(128)              // Only for simulation. Should be reconsidered.
    ) controller_inst(
        // Global Signals
        .clk(clk),
        .rstn(rstn),
        // Global Config Signals
        .gbus_addr(gbus_addr),
        .gbus_wen(gbus_wen),
        .gbus_ren(gbus_ren),

        .vlink_enable(vlink_enable),
        .vlink_wen(vlink_wen),

        //hlink_wdata go through reg, to hlink_rdata
        .hlink_wen(hlink_wen),

        //Global SRAM Access Bus
        .global_sram_waddr(global_sram_waddr),
        .global_sram_raddr(global_sram_raddr),
        .global_sram_wen(global_sram_wen),
        .global_sram_ren(global_sram_ren),
        .global_sram_wsel(global_sram_wsel),
        .global_sram_rsel(global_sram_rsel),

        // Channel - MAC Operation
        // Core Memory Access for Weight and KV Cache
        .arr_cmem(arr_cmem),
        // Local Buffer Access for Weight and KV Cache
        .lbuf_empty(lbuf_empty),
        .lbuf_full(lbuf_full),
        .lbuf_ren(lbuf_ren),
        // Local Buffer Access for Activation
        .abuf_empty(abuf_empty),
        .abuf_full(abuf_full),
        .abuf_ren(abuf_ren),

        .gbus_rvalid(gbus_rvalid),
        .ctrl_cons_ovalid(ctrl_cons_ovalid)
    );

    // Dump waveform file
    initial begin
        // $sdf_annotate("controller.syn.sdf", controller_inst);
	    $dumpfile("controller_testbench.dump"); 
        // $fsdbDumpfile("controller_testbench.fsdb");
        $dumpvars(0, controller_testbench);
        // $fsdbDumpvars(0, controller_testbench);
    end

    // Reset
    task reset_task(input int RESET_TIME);
        clk = 0;
        rstn = 0;
        lbuf_empty = '0;
        lbuf_full = '0;
        abuf_empty = '0;
        abuf_full = '0;
        gbus_rvalid = '0;
        # RESET_TIME;
        rstn = 1;
    endtask

    // Finish simulation
    task finish_task(input int FINISH_TIME);
        repeat(FINISH_TIME) begin
            @(posedge clk);
        end
        $display("Finish simulation successfully!");
        $finish;
    endtask

    // Stimuli generation
    task stimuli_task;
        controller_inst.inst_reg=ATT;
        ctrl_cons_ovalid = {`ARR_HNUM{1'b1}};
        gbus_rvalid[`ARR_HNUM-1] = {`ARR_VNUM{1'b1}};
        gbus_rvalid[0] = {`ARR_VNUM{1'b1}};
    endtask



    initial begin

        reset_task(20);

        stimuli_task;

        finish_task(20000);

    end

    // Clock signal
    always #0.5 clk = ~clk;

endmodule


