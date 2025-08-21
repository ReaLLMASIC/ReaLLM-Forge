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

    logic                                             vlink_enable;
    logic       [`ARR_VNUM-1:0][`ARR_GBUS_DATA-1:0]   vlink_wdata;
    logic       [`ARR_VNUM-1:0]                       vlink_wen;

    logic       [`ARR_HNUM-1:0][`ARR_GBUS_DATA-1:0]   hlink_wdata;
    logic       [`ARR_HNUM-1:0]                       hlink_wen;

    logic       [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0]    global_sram_waddr;
    logic       [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0]    global_sram_raddr;
    logic                                             global_sram_wen;
    logic                                             global_sram_ren;
    GSRAM_WSEL                                        global_sram_wsel;
    GSRAM_RSEL                                        global_sram_rsel;

    CMEM_ARR_PACKET                                   arr_cmem;

    CTRL                                              lbuf_empty;
    CTRL                                              lbuf_full;
    CTRL                                              lbuf_ren;
    CTRL                                              lbuf_reuse_ren;
    CTRL                                              lbuf_reuse_rst;

    CTRL                                              abuf_empty;
    CTRL                                              abuf_full;
    CTRL                                              abuf_ren;
    CTRL                                              abuf_reuse_ren;
    CTRL                                              abuf_reuse_rst;


    CTRL                                              gbus_rvalid;
    CTRL                                              gbus_rvalid_tmp;

    // ============ For simulation =============
    logic abuf_kcompute_raddr_inc;

    // =========================================

    controller #(
        .INST_REG_DEPTH(128)              // Only for simulation. Should be reconsidered.
    ) controller_inst(
        .*
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
        lbuf_empty = '1;
        lbuf_full = '0;
        
        abuf_empty = '1;
        abuf_full = '0;

        gbus_rvalid = '0;
        gbus_rvalid_tmp = '0;
    
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
    task load_task;

    endtask



    initial begin

        reset_task(20);

        finish_task(500000);

    end

    // Generating abuf_empty and lbuf_empty signal
    always@(posedge clk) begin
        if(abuf_kcompute_raddr_inc) begin
            # 2;
            for(int i=0;i<`ARR_HNUM;i++) begin
                abuf_empty[i][0] = 0;
                lbuf_empty[i][0] = 0;
            end
            for(int i=1;i<`ARR_VNUM;i++) begin
                # 1;
                for(int j=0;j<`ARR_HNUM;j++) begin
                    abuf_empty[j][i] = abuf_empty[j][i-1];
                    lbuf_empty[j][i] = lbuf_empty[j][i-1];
                end
            end
        end
        else begin
            for(int i=0;i<`ARR_HNUM;i++) begin
                abuf_empty[i][0] = 1;
                lbuf_empty[i][0] = 1;
            end
            for(int i=1;i<`ARR_VNUM;i++) begin
                # 1;
                for(int j=0;j<`ARR_HNUM;j++) begin
                    abuf_empty[j][i] = abuf_empty[j][i-1];
                    lbuf_empty[j][i] = lbuf_empty[j][i-1];
                end
            end
        end
    end


    // Generating rvalid signal

    always@(posedge abuf_kcompute_raddr_inc) begin
        # 8;
        repeat(6) begin
            repeat(2) begin
                for(int cnt=0; cnt<=`ARR_VNUM; cnt++) begin
                    if(cnt == `ARR_VNUM)
                        gbus_rvalid = '0;
                    else begin
                        for(int i=0;i<`ARR_HNUM;i++) begin
                            for(int j=0; j<`ARR_VNUM; j++) begin
                                if(j == cnt)
                                    gbus_rvalid[i][j] = 1;
                                else
                                    gbus_rvalid[i][j] = 0;
                            end
                        end
                        # 1;
                        for(int i=0;i<`ARR_HNUM;i++) begin
                            gbus_rvalid[i][cnt] = 0;
                        end
                        #3;
                    end
                end
            end
        end
    end

    // Generating clock signal
    always #0.5 clk = ~clk;

endmodule

