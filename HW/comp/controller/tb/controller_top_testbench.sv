
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


module controller_top_testbench #(

);

    logic                                           clk;
    logic                                           rstn;
    // Global Config Signals
    CFG_ARR_PACKET                                  arr_cfg;
    // Channel - Global Bus to Access Core Memory and MAC Result
    // 1. Write Channel
    //      1.1 Chip Interface -> WMEM for Weight Upload
    //      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
    //      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
    // 2. Read Channel
    //      2.1 WMEM       -> Chip Interface for Weight Check
    //      2.2 KV Cache   -> Chip Interface for KV Checnk
    //      2.3 MAC Result -> Vector Engine  for Post Processing
    logic   [`N_HEAD-1:0][`ARR_GBUS_ADDR-1:0]       gbus_addr;
    CTRL                                            gbus_wen;
    logic   [`N_HEAD-1:0][`ARR_GBUS_DATA-1:0]       gbus_wdata;     // From Global SRAM for weight loading
    CTRL                                            gbus_ren;
    logic   [`N_HEAD-1:0][`ARR_GBUS_DATA-1:0]       gbus_rdata;     // To Chip Interface (Debugging) and Vector Engine (MAC)
    CTRL                                            gbus_rvalid;
    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    logic                                           vlink_enable;
    logic   [`ARR_VNUM-1:0][`ARR_GBUS_DATA-1:0]     vlink_wdata;
    logic   [`ARR_VNUM-1:0]                         vlink_wen;
    logic   [`ARR_VNUM-1:0][`ARR_GBUS_DATA-1:0]     vlink_rdata;
    logic   [`ARR_VNUM-1:0]                         vlink_rvalid;
    // Horizontal for Activation Propagation
    logic   [`ARR_HNUM-1:0][`ARR_GBUS_DATA-1:0]     hlink_wdata;    //hlink_wdata go through reg, to hlink_rdata
    logic   [`ARR_HNUM-1:0]                         hlink_wen;   
    logic   [`ARR_HNUM-1:0][`ARR_GBUS_DATA-1:0]     hlink_rdata;
    logic   [`ARR_HNUM-1:0]                         hlink_rvalid;
    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    CMEM_ARR_PACKET                                 arr_cmem;
    // Local Buffer Access for Weight and KV Cache
    CTRL                                            lbuf_empty;
    CTRL                                            lbuf_reuse_empty;
    CTRL                                            lbuf_reuse_ren; //reuse pointer logic, when enable
    CTRL                                            lbuf_reuse_rst;  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    CTRL                                            lbuf_full;
    CTRL                                            lbuf_almost_full;
    CTRL                                            lbuf_ren;
    // Local Buffer Access for Activation
    CTRL                                            abuf_empty;
    CTRL                                            abuf_reuse_empty;
    CTRL                                            abuf_reuse_ren; //reuse pointer logic, when enable
    CTRL                                            abuf_reuse_rst;  //reuse reset logic, when first round of reset is finished, reset reuse pointer to current normal read pointer value
    CTRL                                            abuf_full;
    CTRL                                            abuf_almost_full;
    CTRL                                            abuf_ren;

    logic   [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram_waddr;
    logic   [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram_raddr;
    logic                                           global_sram_wen;
    logic                                           global_sram_ren;
    GSRAM_WSEL                                      global_sram_wsel;
    GSRAM_RSEL                                      global_sram_rsel;

    logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram0_waddr;
    logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram0_raddr;
    logic                                             global_sram0_wen;
    logic                                             global_sram0_ren;
    GSRAM_WSEL                                        global_sram0_wsel;
    GSRAM_RSEL                                        global_sram0_rsel;

    //Mux select signals
    HLINK_WSEL                                        hlink_sel;
    GBUS_WSEL                                         gbus_sel;
    LN_WSEL                                           ln_sel;
    //vec engine valid
    logic [`ARR_HNUM-1:0]                             ctrl_cons_valid;
    logic                                             ctrl_ln_valid;
    logic [`ARR_HNUM-1:0]                             ctrl_wb_valid;
    //SFR connection
    logic                                             start;
    logic [$clog2(8)-1:0]                             inst_reg_addr;
    logic                                             inst_reg_wen;
    logic                                             inst_reg_ren;
    GPT_COMMAND                                       inst_reg_wdata;
    GPT_COMMAND                                       inst_reg_rdata;

    logic [`ARR_GBUS_DATA-1:0]                        random_number;
    logic [`ARR_HNUM-1:0]                             ctrl_cons_ovalid;
    logic  [$clog2(8)-1:0]                            pc_reg;
    


    core_array #(
        .H_NUM(`ARR_HNUM),
        .V_NUM(`ARR_VNUM),

        .GBUS_DATA(`ARR_GBUS_DATA),
        .GBUS_ADDR(`ARR_GBUS_ADDR),

        .LBUF_DEPTH(`ARR_LBUF_DEPTH),
        .LBUF_DATA(`ARR_LBUF_DATA),
        .LBUF_ADDR($clog2(`ARR_LBUF_DEPTH)),

        .CDATA_BIT(`ARR_CDATA_BIT),
        .ODATA_BIT(`ARR_ODATA_BIT),
        .IDATA_BIT(`ARR_IDATA_BIT),
        .MAC_NUM  (`ARR_MAC_NUM),

        .WMEM_DEPTH(`ARR_WMEM_DEPTH),             // WMEM Size
        .CACHE_DEPTH(`ARR_CACHE_DEPTH)              // KV Cache Size

    ) core_array_inst(
        .*
    );

    controller #(
        .INST_REG_DEPTH(8)              // Only for simulation, need to be reconsidered
    ) controller_inst(
        .*
    );

// Dump waveform file
    initial begin
        // $sdf_annotate("../../../syn/data/controller.syn.sdf", controller_inst,,,"MAXIMUM");
	    $dumpfile("controller_top_testbench.dump"); 
        // $fsdbDumpfile("./controller_top_testbench.fsdb");
        $dumpvars(0, controller_top_testbench);
        // $fsdbDumpvars();
    end

    // Reset
    task reset_task(input int RESET_TIME);
        clk = 0;
        rstn = 0;
        gbus_wdata = '0;
        hlink_wdata = '0;
        vlink_wdata = '0;
    
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
        random_number = $urandom();
        for(int i = 0; i < `N_HEAD; i++) begin
            hlink_wdata[i] = random_number;
        end
        for(int i = 0; i < `N_HEAD; i++) begin
            gbus_wdata[i] = random_number;
        end
    endtask

    initial begin

        reset_task(20);

        repeat(70000) begin
            stimuli_task;
            #1;
        end

        finish_task(20);

    end

    // Generating clock signal
    always #0.5 clk = ~clk;


endmodule
