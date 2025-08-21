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


//FIXME: cmem don't need to connect to all the cores, connect like hlink.
//FIXME: abuf and lbuf full or empty don't need all the cores, only first column.
//FIXME: abuf and lbuf full can only have one.
//FIXME: The write back of Q gen should be written back to different SRAMs, so that during attention each head's Q can be read at same time.
//FIXME: 1 Global SRAM storing input activation should not be overwritten by ping-pong mechanism, because it might be used by residual path.
//FIXME: Q and S read to abuf and lbuf can be optimized, now After Q0 and S0 has been filled in, next Q for first head won't start fill until other head's first S has finished filling. （att_saddr_overflow is controled by last head's s read）
`include "../comp/sys_defs.svh"
module controller #(
    parameter INST_REG_DEPTH = 128              // Only for simulation, need to be reconsidered
)(
    // Global Signals
    input                       clk,
    input                       rstn,
    //program counter todo
    output logic [$clog2(INST_REG_DEPTH)-1:0] pc_reg,
    // Channel - Global Bus to Access Core Memory and MAC Result
    // 1. Write Channel
    //      1.1 Chip Interface -> WMEM for Weight Upload
    //      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
    //      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
    // 2. Read Channel
    //      2.1 WMEM       -> Chip Interface for Weight Check
    //      2.2 KV Cache   -> Chip Interface for KV Check
    //      2.3 MAC Result -> Vector Engine  for Post Processing
    output logic       [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-1:0]       gbus_addr,
    output CTRL                                                  gbus_wen,
    output CTRL                                                  gbus_ren,
    input  CTRL                                                  gbus_rvalid,

    output logic                                             vlink_enable,
    // output logic       [`ARR_VNUM-1:0][`ARR_GBUS_DATA-1:0]   vlink_wdata,
    output logic       [`ARR_VNUM-1:0]                       vlink_wen,
    input              [`ARR_VNUM-1:0]                       vlink_rvalid,

    // output logic       [`ARR_HNUM-1:0][`ARR_GBUS_DATA-1:0]   hlink_wdata,    //hlink_wdata go through reg, to hlink_rdata
    output logic       [`ARR_HNUM-1:0]                       hlink_wen,
    input              [`ARR_HNUM-1:0]                       hlink_rvalid,

    //Global SRAM0 Access Bus
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]    global_sram_waddr,
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]    global_sram_raddr,
    output logic                                             global_sram_wen,
    output logic                                             global_sram_ren,
    output GSRAM_WSEL                                        global_sram_wsel,
    output GSRAM_RSEL                                        global_sram_rsel,

    //Global SRAM1 Access Bus
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram0_waddr,
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram0_raddr,
    output logic                                             global_sram0_wen,
    output logic                                             global_sram0_ren,
    output GSRAM_WSEL                                        global_sram0_wsel,
    output GSRAM_RSEL                                        global_sram0_rsel,

    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    output CMEM_ARR_PACKET  arr_cmem,
    // Local Buffer Access for Weight and KV Cache
    input  CTRL             lbuf_empty,
    input  CTRL             lbuf_full,
    output CTRL             lbuf_ren,
    output CTRL             lbuf_reuse_ren,
    output CTRL             lbuf_reuse_rst,
    // Local Buffer Access for Activation
    input  CTRL             abuf_empty,
    input  CTRL             abuf_full,
    output CTRL             abuf_reuse_ren,
    output CTRL             abuf_reuse_rst,
    output CTRL             abuf_ren,
    //Mux select signals
    output HLINK_WSEL       hlink_sel,
    output GBUS_WSEL        gbus_sel,
    output LN_WSEL          ln_sel,
    //vec engine valid
    output logic [`ARR_HNUM-1:0] ctrl_cons_valid,
    output logic            ctrl_ln_valid,
    output logic [`ARR_HNUM-1:0] ctrl_wb_valid,
    //SFR connection
    input                   start,
    input [$clog2(INST_REG_DEPTH)-1:0] inst_reg_addr,
    input                   inst_reg_wen,
    input                   inst_reg_ren,
    input  GPT_COMMAND      inst_reg_wdata,
    output GPT_COMMAND      inst_reg_rdata,

    //From Consmax
    input [`ARR_HNUM-1:0] ctrl_cons_ovalid
);

/************* INST_REG *************/
GPT_COMMAND [INST_REG_DEPTH-1:0] inst_reg ;
STATE state,next_state;

//finishing flags for fsm
logic load_finish, compute_finish;

//global sram pingpong
logic pingpong_reg,next_pingpong_reg;
/************* VGEN LOAD_WEIGHT *************/
parameter BIT_WIDTH = $clog2(`ARR_MAC_NUM);
//counters
logic in_core_inc;
logic in_core_inc_d;
logic in_core_overflow;
logic [BIT_WIDTH-1:0] in_core_cnt;

logic core_col_overflow;
logic [BIT_WIDTH-1:0] core_col_cnt;

logic core_row_overflow;
logic [BIT_WIDTH-1:0] core_row_cnt;

//load weight for QKV gen
//each core stalls n_model/n_head/n_head channel, each channel is n_model length.
//gbus_data is ARR_GBUS_DATA bits, each data is ARR_IDATA_BIT bits, a word line can save ARR_GBUS_DATA/ARR_IDATA_BIT number of data
//each core need n_model/n_head/n_head * n_model/(ARR_GBUS_DATA/ARR_IDATA_BIT) address
//first load the column (one head), then load the row (different heads)
//global sram read address accumulate by 1 each cycle. 
localparam CORE_ADDR_CNT = `N_MODEL*`N_MODEL/`N_HEAD/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0] gbus_load_vgen_addr; //gbus_load_vgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0] global_sram_load_vgen_addr;//extra 1 bit for differentiating writing to kv cache or wmem                                             
CTRL                       gbus_load_vgen_wen;

/************* VGEN Compute Registers *************/
//load activation for abuf through Hlink
localparam ABUF_CNT = `N_MODEL*`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);

logic                                                       vcompute_core_en;

CTRL                                                        cmem_vcompute_ren;
logic [`ARR_VNUM-1:0]                                       cmem_vcompute_ren_reg;
logic [`ARR_VNUM-1:0][`ARR_GBUS_ADDR-1:0]                   cmem_vcompute_raddr_reg;
G_ADDR                                                      cmem_vcompute_raddr;

logic [`ARR_HNUM-1:0]                                       hlink_vcompute_wen;

G_ADDR                                                      vcompute_cmem_waddr;      // Write Value to KV Cache, G_BUS -> KV Cache, debug.
CTRL                                                        vcompute_cmem_wen;
logic [`ARR_GBUS_ADDR-1:0]                                  cmem_write_vcompute_addr; //todo 3_27 what is this.

logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                      global_sram_compute_vgen_raddr;
logic                                                       global_sram_compute_vgen_ren;
logic                                                       global_sram_compute_vgen_ren_temp;

logic [`ARR_VNUM-1:0]                                       labuf_compute_vgen_valid;
logic                                                       gnt_compute_vgen_en;
logic [`ARR_VNUM-1:0]                                       labuf_compute_vgen_ren;

CTRL                                                        abuf_compute_vgen_ren;
CTRL                                                        lbuf_compute_vgen_ren;

/******QGEN loadweight gbus_addr generation******/
logic qload_finish, qcompute_finish;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0] gbus_load_qgen_addr; //gbus_load_qgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0] global_sram_load_qgen_addr;//extra 1 bit for differentiating writing to kv cache or wmem                                             
CTRL                       gbus_load_qgen_wen;

/*************Q LOAD_WEIGHT counter *************/
logic qload_core_inc;
logic qload_core_inc_d;
logic qload_in_core_overflow;
logic [BIT_WIDTH-1:0] qload_in_core_cnt;

logic qload_core_col_overflow;
logic [BIT_WIDTH-1:0] qload_core_col_cnt;

logic qload_core_row_overflow;
logic [BIT_WIDTH-1:0] qload_core_row_cnt;


/************* QGEN Compute Registers *************/
logic [`ARR_GBUS_ADDR-1:0]                                  cmem_read_qcompute_addr;
logic                                                       qcompute_core_inc;
logic                                                       qcompute_core_inc_d;
CTRL                                                        cmem_qcompute_ren;
G_ADDR                                                      cmem_qcompute_raddr;
logic                                                       qcompute_core_overflow;
logic [BIT_WIDTH-1:0]                                       qcompute_core_cnt;

CTRL                                                        qcompute_abuf_ren;
CTRL                                                        qcompute_lbuf_ren;
CTRL                                                        qcompute_gbus_ren;

logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0]                    global_sram_compute_qgen_waddr;
logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                      global_sram_compute_qgen_addr;


/******KGEN loadweight gbus_addr generation******/
logic kload_finish, kcompute_finish;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0] gbus_load_kgen_addr; //gbus_load_kgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0] global_sram_load_kgen_addr;//extra 1 bit for differentiating writing to kv cache or wmem                                             
CTRL                       gbus_load_kgen_wen;

/*************K LOAD_WEIGHT counter *************/
logic kload_core_inc;
logic kload_core_inc_d;
logic kload_in_core_overflow;
logic [BIT_WIDTH-1:0] kload_in_core_cnt;

logic kload_core_col_overflow;
logic [BIT_WIDTH-1:0] kload_core_col_cnt;

logic kload_core_row_overflow;
logic [BIT_WIDTH-1:0] kload_core_row_cnt;

/************* KGEN Compute Registers *************/
logic [`ARR_VNUM-1:0][`ARR_GBUS_ADDR-1:0]                   cmem_compute_kgen_raddr_1D;
CTRL                                                        cmem_compute_kgen_ren;
G_ADDR                                                      cmem_compute_kgen_raddr;
CTRL                                                        abuf_compute_kgen_ren;
CTRL                                                        lbuf_compute_kgen_ren;
G_ADDR                                                      kcompute_cmem_waddr;      // Write Value to KV Cache, G_BUS -> KV Cache, debug.
CTRL                                                        kcompute_cmem_wen;
logic [`ARR_GBUS_ADDR-1:0]                                  cmem_write_kcompute_addr;
logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                      global_sram_compute_kgen_addr;
logic                                                       global_sram_compute_kgen_ren;
logic [`ARR_HNUM-1:0]                                       hlink_compute_kgen_wen;
logic [`ARR_VNUM-1:0]      labuf_compute_kgen_valid;
logic                      gnt_compute_kgen_en;
logic [`ARR_VNUM-1:0]      labuf_compute_kgen_ren;


    //////////////////////////////////////////////////
    //                                              //
    //       FSM                                    //
    //                                              //
    //////////////////////////////////////////////////

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        pingpong_reg<=0;
    end
    else if(inst_reg[pc_reg]==ATT) begin
        pingpong_reg<=next_pingpong_reg; //continue here
    end
end
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        state<=LOAD_WEIGHT;
    end
    else begin
        state<=next_state;
    end
end
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        pc_reg<='0;
    end
    else begin
        //TODO:
    end
end
always_comb begin
    case(state)
        LOAD_WEIGHT: begin
            next_state = load_finish ? COMPUTE : LOAD_WEIGHT;
            next_pingpong_reg = pingpong_reg;
        end
        COMPUTE: begin
            next_state = compute_finish ? FINISH : COMPUTE;
            next_pingpong_reg = pingpong_reg;
        end
        FINISH: begin
            next_state = LOAD_WEIGHT;
            next_pingpong_reg = ~pingpong_reg;
        end
        default: begin
            next_state = LOAD_WEIGHT;
            next_pingpong_reg = pingpong_reg;
        end
    endcase
end

    //////////////////////////////////////////////////
    //                                              //
    //       Global Control Signal Select           //
    //                                              //
    //////////////////////////////////////////////////
//FIXME: After going through this mux, better reg it.
always_comb begin
    //initialize
    load_finish = 1'b0;
    compute_finish = 1'b0;
    vlink_enable   = 1'b0;
    vlink_wen      = 1'b0;

    gbus_addr='0;
    gbus_wen ='0;
    gbus_ren ='0;

    arr_cmem.cmem_raddr = '0;
    arr_cmem.cmem_ren = '0;
    arr_cmem.cmem_wen = '0;
    arr_cmem.cmem_waddr = '0;

    hlink_wen = '0;

    global_sram_raddr = '0;
    global_sram_ren   = '0;
    global_sram_rsel = RSEL_DISABLE;
    global_sram_wen = '0;
    global_sram_waddr = '0;
    global_sram_wsel = WSEL_DISABLE;

    global_sram0_ren = '0;
    global_sram0_raddr = '0;
    global_sram0_rsel = RSEL_DISABLE;
    global_sram0_waddr = '0;
    global_sram0_wen = '0;
    global_sram0_wsel = WSEL_DISABLE;

    lbuf_ren = '0;
    lbuf_reuse_ren = '0;
    lbuf_reuse_rst = '0;

    abuf_ren = '0;
    abuf_reuse_ren = '0;
    abuf_reuse_rst = '0;


    case(inst_reg[pc_reg]) 
        ATT: begin
            load_finish = (state == LOAD_WEIGHT)        ?   1                               : '0;
            compute_finish = (state == COMPUTE )        ?   att_wb_finish                   : '0;
            arr_cmem       = (state == COMPUTE )        ?   att_cmem                        : '0;
            hlink_wen      = (state == COMPUTE)         ?   '1                              : '0;

            global_sram0_raddr = (state == COMPUTE)      ?   global_sram_att_q_raddr         : '0;
            global_sram0_ren   = (state == COMPUTE)      ?   global_sram_att_q_ren           : '0;
            global_sram0_rsel = (state == LOAD_WEIGHT)   ?   global_sram_att_q_rsel          : '0;

            global_sram_wsel  = ;//FIXME

            lbuf_ren = (state == COMPUTE)               ?   qcompute_lbuf_ren               : '0;
            abuf_ren = (state == COMPUTE)               ?   qcompute_abuf_ren               : '0;
        end
    endcase
end

    //////////////////////////////////////////////////
    //                                              //
    //                Attention                     //
    //                                              //
    //////////////////////////////////////////////////
//FIXME: compute_finish signal, ctrl_cons_valid,
assign compute_finish=1'b0;
localparam GSRAM_ATT_QADDR_CNT = `N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
localparam VEC_ATT_SADDR_CNT = `SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
localparam LBUF_ATT_CMEM_CNT = `SEQ_LENGTH/(`ARR_VNUM)*`N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
//attention counters
logic [$clog2(`N_HEAD)-1:0] att_head_cnt;
logic att_head_overflow;
logic att_head_inc;
logic [$clog2(GSRAM_ATT_QADDR_CNT)-1:0] att_qaddr_cnt;
logic att_qaddr_overflow;
logic att_qaddr_inc;
logic [$clog2(VEC_ATT_SADDR_CNT)-1:0] att_saddr_cnt;
logic att_saddr_overflow;
logic att_saddr_inc;
logic [$clog2(`SEQ_LENGTH)-1:0] att_seq_cnt;
logic att_seq_overflow;
logic att_seq_inc;
//attention global sram read
logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0] global_sram_att_q_raddr;
logic global_sram_att_q_ren;
GSRAM_RSEL global_sram_att_q_rsel;
//attention flags
logic att_en;
logic att_interleave_flag,next_att_interleave_flag;
logic att_rd_finish,next_att_rd_finish;
assign att_en = (inst_reg[pc_reg]==ATT) && (state==COMPUTE);
//attention hlink
logic [`ARR_HNUM-1:0] hlink_att_wen,next_hlink_att_wen;
HLINK_WSEL att_hlink_sel,next_att_hlink_sel;
//attention cmem
CMEM_ARR_PACKET att_cmem;

/******ABUF: Read Q from Global SRAM, Read P interleaving******/
//counter increment logic
always_comb begin
    att_head_inc = 1'b0;
    next_hlink_att_wen = 'b0;
    next_att_hlink_sel = att_hlink_sel;
    if(att_en & ~att_rd_finish) begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            if(~att_interleave_flag) begin// qaddr head count
                if(att_head_cnt==i) begin
                    if(~abuf_full[i][0]) begin//head's abuf is not full
                        att_head_inc=1'b1;
                        next_hlink_att_wen[i] = 1'b1;
                        next_att_hlink_sel = GSRAM02HLINK;
                    end
                end
            end
            else begin// saddr: write back from consmax, using last head's consmax output valid signal as indicator
                if(~abuf_full[i][0] & ctrl_cons_ovalid[i]) begin //FIXME: When abuf full, we will lose the data from consmax
                    next_att_hlink_sel = CONS2HLINK;
                    next_hlink_att_wen[i] = 1'b1;
                end
            end
        end
    end
end

assign att_qaddr_inc = att_head_overflow & ~att_interleave_flag & ~att_rd_finish;
assign att_saddr_inc = ctrl_cons_ovalid[`ARR_HNUM-1] & att_interleave_flag & ~att_rd_finish;
assign att_seq_inc   = att_saddr_overflow & att_interleave_flag & ~att_rd_finish;

//Q,P interleave read flag
always_comb begin
    next_att_interleave_flag=att_interleave_flag;
    next_att_rd_finish=1'b1;
    if(att_en) begin
        if(~att_rd_finish) begin
            if(att_qaddr_overflow | att_saddr_overflow) begin  
                next_att_interleave_flag = ~att_interleave_flag;
            end
            if(~att_seq_overflow) begin
                next_att_rd_finish=1'b0;
            end
        end
    end
    else begin
        next_att_rd_finish = 1'b0; //reset attention load finish flag when attention layer finish
    end
end

always_comb begin
    global_sram_att_q_raddr='0;
    global_sram_att_q_ren = 1'b0;
    global_sram_att_q_rsel = GSRAM2CHIP;
    if(att_en) begin
        global_sram_att_q_rsel = GSRAM2CHIP;
        for(int i=0;i<`ARR_HNUM;i++) begin
            if(att_head_cnt==i) begin
                if(~abuf_full[i][0] & ~att_interleave_flag) begin//head's abuf is not full
                    global_sram_att_q_ren = 1'b1;
                    global_sram_att_q_raddr = `GSRAM_ATT_QBASE_ADDR + att_head_cnt + att_qaddr_cnt * `N_HEAD + att_seq_cnt * `N_HEAD * GSRAM_ATT_QADDR_CNT;
                end
            end
        end
    end
end

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_interleave_flag<=1'b0;
        att_rd_finish<=1'b0;
        hlink_att_wen<='b0;
        att_hlink_sel<='b0;
    end
    else if(att_en) begin
        att_interleave_flag<=next_att_interleave_flag;
        att_rd_finish<=next_att_rd_finish;
        hlink_att_wen<=next_hlink_att_wen;
        att_hlink_sel<=next_att_hlink_sel;
    end
end

/******LBUF: Read K from KV CACHE (CMEM), Read V interleaving******/
//To save power, only write to lbuf when abuf writes.
//cmem raddr should be same as q addr.
always_comb begin
    att_cmem = '0;
    if(att_en & ~att_rd_finish) begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            if(~att_interleave_flag) begin
                if(att_head_cnt==i) begin
                    //cmem_raddr for K, (Q K)
                    if(~abuf_full[i][0]) begin
                        for(int j=0;j<`ARR_VNUM;j++) begin
                            att_cmem.cmem_ren[i][j] = 1'b1;
                            att_cmem.cmem_raddr[i][j] = att_qaddr_cnt + att_seq_cnt*GSRAM_ATT_QADDR_CNT + `CMEM_KBASE_ADDR;
                        end
                    end
                end
            end
            else if(ctrl_cons_ovalid[i]) begin
                //cmem_raddr for V, (P V)
                if(~abuf_full[0][0] & att_interleave_flag) begin
                    for(int j=0;j<`ARR_VNUM;j++) begin
                        att_cmem.cmem_ren[i][j] = 1'b1;
                        att_cmem.cmem_raddr[i][j] = att_saddr_cnt + att_seq_cnt*VEC_ATT_SADDR_CNT + `CMEM_VBASE_ADDR;
                    end
                end
            end
        end
    end
end

/******Compute: ABUF Reuse and Ren, LBUF Ren******/
//Attention QK reuse control
logic att_q_abuf_reuse_finish;
CTRL  att_q_abuf_ren;
CTRL  att_q_abuf_ren_reg;
CTRL  att_q_abuf_reuse_ren;
CTRL  att_q_abuf_reuse_ren_reg;
CTRL  att_q_abuf_reuse_rst;
CTRL  att_q_abuf_reuse_rst_reg;
CTRL  att_k_lbuf_ren;
CTRL  att_k_lbuf_ren_reg;
CTRL  att_k_lbuf_reuse_ren;
CTRL  att_k_lbuf_reuse_rst;
//Attention PV reuse control
logic att_p_abuf_reuse_finish;
CTRL  att_p_abuf_ren;
CTRL  att_p_abuf_ren_reg;
CTRL  att_p_abuf_reuse_ren;
CTRL  att_p_abuf_reuse_ren_reg;
CTRL  att_p_abuf_reuse_rst;
CTRL  att_p_abuf_reuse_rst_reg;
CTRL  att_v_lbuf_ren;
CTRL  att_v_lbuf_ren_reg;
CTRL  att_v_lbuf_reuse_ren;
CTRL  att_v_lbuf_reuse_rst;
//Attention QK reuse counter
//Max count for each Loop for Q reuse during QK
localparam ATT_QK_CNT = `N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
//Max count for loop
localparam ATT_QK_LOOP = `SEQ_LENGTH/(`ARR_VNUM);
logic                                      att_qk_cnt_overflow;
logic [$clog2(ATT_QK_CNT)-1:0]           att_qk_cnt;
logic                                      att_qk_inc;
logic                                      att_qk_reuse_loop_overflow;
logic [$clog2(ATT_QK_LOOP)-1:0]     att_qk_reuse_loop_cnt;
logic                                      att_qk_reuse_loop_inc;
//Attention PV reuse counter
//Max count for each Loop for P reuse during PV
localparam ATT_PV_CNT = `SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
//Max count for loop
localparam ATT_PV_LOOP = `N_MODEL/`N_HEAD/(`ARR_VNUM);
logic                                      att_pv_cnt_overflow;
logic [$clog2(ATT_PV_CNT)-1:0]             att_pv_cnt;
logic                                      att_pv_inc;
logic                                      att_pv_reuse_loop_overflow;
logic [$clog2(ATT_PV_LOOP)-1:0]            att_pv_reuse_loop_cnt;
logic                                      att_pv_reuse_loop_inc;
//compute sequence counter
logic [$clog2(`SEQ_LENGTH)-1:0] att_compute_seq_cnt;
logic att_compute_seq_overflow;
logic att_compute_seq_inc;
//Attention QK, PV interleave flag
logic att_compute_interleave_flag;
//Attention compute finish flag
logic att_compute_finish, next_att_compute_finish;
/******Attention QK Logic******/
assign att_qk_inc = (!abuf_empty[0][0]) && (!lbuf_empty[0][0]) && ~att_compute_interleave_flag && att_en && ~att_compute_finish; //att_en start signal, att_compute_finish is finish signal 
assign att_qk_reuse_loop_inc = att_qk_cnt_overflow;
assign att_q_abuf_reuse_finish = (att_qk_reuse_loop_cnt == ATT_QK_LOOP-1);

generate
    for(genvar i=0; i<`ARR_HNUM; i++) begin
        for(genvar j=0;j<`ARR_HNUM;j++) begin
            if(i==0 & j==0) begin
                assign att_q_abuf_ren[0][0] = att_q_abuf_reuse_finish && (!abuf_empty[0][0]) && (!lbuf_empty[0][0]);
                assign att_q_abuf_reuse_ren[0][0] = (!att_q_abuf_reuse_finish) && (!abuf_empty[0][0]) && (!lbuf_empty[0][0]) && ~att_compute_interleave_flag && att_en && ~att_compute_finish;
                assign att_q_abuf_reuse_rst[0][0] = ((att_qk_reuse_loop_cnt == ATT_QK_LOOP-2) && (att_qk_cnt == ATT_QK_CNT-1) && att_q_abuf_reuse_ren[0][0]);
                assign att_k_lbuf_ren[0][0] = att_q_abuf_ren[0][0] || att_q_abuf_reuse_ren[0][0];
            end
            else begin
                assign att_q_abuf_ren[i][j] = att_q_abuf_ren_reg[i][j];
                assign att_q_abuf_reuse_ren[i][j] = att_q_abuf_reuse_ren_reg[i][j];
                assign att_q_abuf_reuse_rst[i][j] = att_q_abuf_reuse_rst_reg[i][j];
                assign att_k_lbuf_ren[i][j] = att_k_lbuf_ren_reg[i][j];
            end
        end
    end
endgenerate

assign att_k_lbuf_reuse_ren = '0;
assign att_k_lbuf_reuse_rst = '0;

//control signal for remaining heads
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=0; i<`ARR_HNUM; i++) begin
            for(int j=0; j<`ARR_VNUM; j++) begin
                att_q_abuf_ren_reg[i][j] <= 0;
                att_k_lbuf_ren_reg[i][j] <= 0;
                att_q_abuf_reuse_ren_reg[i][j] <= 0;
                att_q_abuf_reuse_rst_reg[i][j] <= 0;
            end
        end
    end
    else begin
        //first column
        for(int i=1; i<`ARR_HNUM;i++) begin
            if(i==1) begin
                att_q_abuf_ren_reg[i][0]<=att_q_abuf_ren[0][0];
                att_k_lbuf_ren_reg[i][0] <=att_k_lbuf_ren[0][0];
                att_q_abuf_reuse_ren_reg[i][0] <= att_q_abuf_reuse_ren[0][0];
                att_q_abuf_reuse_rst_reg[i][0] <= att_q_abuf_reuse_rst[0][0];
            end
            else begin
                att_q_abuf_ren_reg[i][0] <= att_q_abuf_ren[i-1][0];
                att_k_lbuf_ren_reg[i][0] <= att_k_lbuf_ren[i-1][0];
                att_q_abuf_reuse_ren_reg[i][0] <= att_q_abuf_reuse_ren[i-1][0];
                att_q_abuf_reuse_rst_reg[i][0] <= att_q_abuf_reuse_rst[i-1][0];
            end
        end
        //other column
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                if(i==1 & j==0) begin
                    att_q_abuf_ren_reg[j][i] <= att_q_abuf_ren[0][0];
                    att_k_lbuf_ren_reg[j][i] <= att_k_lbuf_ren[0][0];
                    att_q_abuf_reuse_ren_reg[j][i] <= att_q_abuf_reuse_ren[0][0];
                    att_q_abuf_reuse_rst_reg[j][i] <= att_q_abuf_reuse_rst[0][0];
                end
                else begin
                    att_q_abuf_ren_reg[j][i] <= att_q_abuf_ren_reg[j][i-1];
                    att_k_lbuf_ren_reg[j][i] <= att_k_lbuf_ren_reg[j][i-1];
                    att_q_abuf_reuse_ren_reg[j][i] <= att_q_abuf_reuse_ren_reg[j][i-1];
                    att_q_abuf_reuse_rst_reg[j][i] <= att_q_abuf_reuse_rst_reg[j][i-1];
                end
            end
        end
    end
end

// Attention compute interleave flag
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_compute_interleave_flag <= 0;
    end
    else if(att_en) begin
        if(att_qk_reuse_loop_overflow | att_pv_reuse_loop_overflow) begin
            att_compute_interleave_flag <= ~att_compute_interleave_flag;
        end
    end
end

/******Attention PV Logic******/
assign att_pv_inc = (!abuf_empty[0][0]) && (!lbuf_empty[0][0]) && att_compute_interleave_flag && att_en && ~att_compute_finish;
assign att_pv_reuse_loop_inc = att_pv_cnt_overflow;
assign att_p_abuf_reuse_finish = (att_pv_reuse_loop_cnt == ATT_PV_LOOP-1);

generate
    for(genvar i=0; i<`ARR_HNUM; i++) begin
        for(genvar j=0;j<`ARR_HNUM;j++) begin
            if(i==0 & j==0) begin
                assign att_p_abuf_ren[0][0] = att_p_abuf_reuse_finish && (!abuf_empty[0][0]) && (!lbuf_empty[0][0]);
                assign att_p_abuf_reuse_ren[0][0] = (!att_p_abuf_reuse_finish) && (!abuf_empty[0][0]) && (!lbuf_empty[0][0]) && att_compute_interleave_flag && att_en && ~att_compute_finish;
                assign att_p_abuf_reuse_rst[0][0] = ((att_pv_reuse_loop_cnt == ATT_PV_LOOP-2) && (att_pv_cnt == ATT_PV_CNT-1) && att_p_abuf_reuse_ren[0][0]);
                assign att_v_lbuf_ren[0][0] = att_p_abuf_ren[0][0] || att_p_abuf_reuse_ren[0][0];
            end
            else begin
                assign att_p_abuf_ren[i][j] = att_p_abuf_ren_reg[i][j];
                assign att_p_abuf_reuse_ren[i][j] = att_p_abuf_reuse_ren_reg[i][j];
                assign att_p_abuf_reuse_rst[i][j] = att_p_abuf_reuse_rst_reg[i][j];
                assign att_v_lbuf_ren[i][j] = att_v_lbuf_ren_reg[i][j];
            end
        end
    end
endgenerate

assign att_v_lbuf_reuse_ren = '0;
assign att_v_lbuf_reuse_rst = '0;

//control signal for remaining heads
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=0; i<`ARR_HNUM; i++) begin
            for(int j=0; j<`ARR_VNUM; j++) begin
                att_p_abuf_ren_reg[i][j] <= 0;
                att_v_lbuf_ren_reg[i][j] <= 0;
                att_p_abuf_reuse_ren_reg[i][j] <= 0;
                att_p_abuf_reuse_rst_reg[i][j] <= 0;
            end
        end
    end
    else begin
        //first column
        for(int i=1; i<`ARR_HNUM;i++) begin
            if(i==1) begin
                att_p_abuf_ren_reg[i][0]<=att_p_abuf_ren[0][0];
                att_v_lbuf_ren_reg[i][0] <=att_v_lbuf_ren[0][0];
                att_p_abuf_reuse_ren_reg[i][0] <= att_p_abuf_reuse_ren[0][0];
                att_p_abuf_reuse_rst_reg[i][0] <= att_p_abuf_reuse_rst[0][0];
            end
            else begin
                att_p_abuf_ren_reg[i][0] <= att_p_abuf_ren[i-1][0];
                att_v_lbuf_ren_reg[i][0] <= att_v_lbuf_ren[i-1][0];
                att_p_abuf_reuse_ren_reg[i][0] <= att_p_abuf_reuse_ren[i-1][0];
                att_p_abuf_reuse_rst_reg[i][0] <= att_p_abuf_reuse_rst[i-1][0];
            end
        end
        //other column
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                if(i==1 & j==0) begin
                    att_p_abuf_ren_reg[j][i] <= att_p_abuf_ren[0][0];
                    att_v_lbuf_ren_reg[j][i] <= att_v_lbuf_ren[0][0];
                    att_p_abuf_reuse_ren_reg[j][i] <= att_p_abuf_reuse_ren[0][0];
                    att_p_abuf_reuse_rst_reg[j][i] <= att_p_abuf_reuse_rst[0][0];
                end
                else begin
                    att_p_abuf_ren_reg[j][i] <= att_p_abuf_ren_reg[j][i-1];
                    att_v_lbuf_ren_reg[j][i] <= att_v_lbuf_ren_reg[j][i-1];
                    att_p_abuf_reuse_ren_reg[j][i] <= att_p_abuf_reuse_ren_reg[j][i-1];
                    att_p_abuf_reuse_rst_reg[j][i] <= att_p_abuf_reuse_rst_reg[j][i-1];
                end
            end
        end
    end
end

assign att_compute_seq_inc = att_pv_reuse_loop_overflow && ~att_compute_finish;

//Attention compute finish flag
always_comb begin
    next_att_compute_finish = 1'b1;
    if(att_en) begin
        if(~att_compute_finish) begin
            if(~att_compute_seq_overflow) begin
                next_att_compute_finish = 1'b0;
            end
        end
    end
    else begin
        next_att_compute_finish = 1'b0; //release finish flag when attention finish
    end
end
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_compute_finish<=0;
    end
    else begin
        att_compute_finish<=next_att_compute_finish;
    end
end

/******Attention Results Writeback******/
localparam ATT_OUT_WB = `N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
//reusing counter for consmax output above to indicate the QK result 
logic [$clog2(ATT_OUT_WB)-1:0] att_wb_out_cnt;
logic att_wb_out_inc;
logic att_wb_out_overflow;
logic [$clog2(VEC_ATT_SADDR_CNT)-1:0] att_cons_out_cnt;
logic att_cons_out_inc;
logic att_cons_out_overflow;
logic [$clog2(`SEQ_LENGTH)-1:0] att_wb_seq_cnt;
logic att_wb_seq_inc;
logic att_wb_seq_overflow;
logic [`ARR_HNUM-2:0]   att_wb_seq_overflow_reg;
logic att_wb_interleave_flag;
logic [`ARR_HNUM-1:0]   att_ctrl_wb_valid;//TODO: ADD TO GLOBAL MUX
logic [`ARR_HNUM-2:0]   att_ctrl_wb_valid_reg;
logic att_ctrl_wb_valid_w;
logic [`ARR_HNUM-1:0]   att_ctrl_cons_valid;//TODO: ADD TO GLOBAL MUX
logic [`ARR_HNUM-2:0]   att_ctrl_cons_valid_reg;
logic att_ctrl_cons_valid_w;
logic att_wb_finish,next_att_wb_finish;

assign att_wb_out_inc = att_wb_interleave_flag & (|gbus_rvalid[0]) & ~att_wb_finish;
assign att_cons_out_inc = ~att_wb_interleave_flag & (|gbus_rvalid[0]) & ~att_wb_finish;
assign att_wb_seq_inc = att_wb_out_overflow;

//write back interleave flag for 1st head
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_wb_interleave_flag <= 0;
    end
    else if(att_en) begin
        if((att_wb_out_overflow)| (att_cons_out_overflow)) begin
            att_wb_interleave_flag <= ~att_wb_interleave_flag;
        end
    end
end
//write back valid for top.
always_comb begin//for 1st head
    att_ctrl_wb_valid_w = 1'b0;
    if(att_en & att_wb_interleave_flag) begin
        att_ctrl_wb_valid_w = 1'b1;
    end
end
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_ctrl_wb_valid_reg <= '0;
    end
    else begin
        for(int i=0;i<`ARR_HNUM-1;i++) begin
            if(i==0) begin
                att_ctrl_wb_valid_reg[i] <= att_ctrl_wb_valid_w;
            end
            else begin
                att_ctrl_wb_valid_reg[i] <= att_ctrl_wb_valid_reg[i-1];
            end
        end
    end
end
assign att_ctrl_wb_valid = {att_ctrl_wb_valid_reg,att_ctrl_wb_valid_w};
//cons valid for top
always_comb begin//for 1st head
    att_ctrl_cons_valid_w = 1'b0;
    if(att_en & ~att_wb_interleave_flag) begin
        att_ctrl_cons_valid_w = 1'b1;
    end
end
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_ctrl_cons_valid_reg <= '0;
    end
    else begin
        for(int i=0;i<`ARR_HNUM-1;i++) begin
            if(i==0) begin
                att_ctrl_cons_valid_reg[i] <= att_ctrl_cons_valid_w;
            end
            else begin
                att_ctrl_cons_valid_reg[i] <= att_ctrl_cons_valid_reg[i-1];
            end
        end
    end
end
assign att_ctrl_cons_valid = {att_ctrl_cons_valid_reg,att_ctrl_cons_valid_w};
//Attention compute finish flag
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_wb_seq_overflow_reg <= '0;
    end
    else begin
        for(int i=0;i<`ARR_HNUM-1;i++) begin
            if(i==0) begin
                att_wb_seq_overflow_reg[i] <= att_wb_seq_overflow;
            end
            else begin
                att_wb_seq_overflow_reg[i] <= att_wb_seq_overflow_reg[i-1];
            end
        end
    end
end
always_comb begin
    next_att_wb_finish = 1'b1;
    if(att_en) begin
        if(~att_wb_finish) begin
            if(~att_wb_seq_overflow_reg[`ARR_HNUM-2]) begin
                next_att_wb_finish = 1'b0;
            end
        end
    end
    else begin
        next_att_wb_finish = 1'b0; //release finish flag when attention finish
    end
end
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_wb_finish<=0;
    end
    else begin
        att_wb_finish<=next_att_wb_finish;
    end
end

counter #(ATT_OUT_WB-1) att_wb_out_counter (.clk(clk),.rstn(rstn),.inc(att_wb_out_inc),.overflow(att_wb_out_overflow),.out(att_wb_out_cnt));
counter #(VEC_ATT_SADDR_CNT-1) att_cons_out_counter (.clk(clk),.rstn(rstn),.inc(att_cons_out_inc),.overflow(att_cons_out_overflow),.out(att_cons_out_cnt));
counter #(`SEQ_LENGTH-1) att_wb_seq_counter (.clk(clk),.rstn(rstn),.inc(att_wb_seq_inc),.overflow(att_wb_seq_overflow),.out(att_wb_seq_cnt));
    //////////////////////////////////////////////////
    //                                              //
    //                Counters                      //
    //                                              //
    //////////////////////////////////////////////////

/*Counters will be merged together once we have all the logic*/
    //Counters for attention
    //abuf and lbuf loading counters
    counter #(`N_HEAD-1) att_head_counter (.clk(clk),.rstn(rstn),.inc(att_head_inc),.overflow(att_head_overflow),.out(att_head_cnt));
    counter #(GSRAM_ATT_QADDR_CNT-1) att_qaddr_counter (.clk(clk),.rstn(rstn),.inc(att_qaddr_inc),.overflow(att_qaddr_overflow),.out(att_qaddr_cnt));
    counter #(VEC_ATT_SADDR_CNT-1) att_saddr_counter (.clk(clk),.rstn(rstn),.inc(att_saddr_inc),.overflow(att_saddr_overflow),.out(att_saddr_cnt));
    counter #(`SEQ_LENGTH-1) att_seq_counter (.clk(clk),.rstn(rstn),.inc(att_seq_inc),.overflow(att_seq_overflow),.out(att_seq_cnt));
    // ATT_QK_CNT: Number of activations in one single loop
    counter #(ATT_QK_CNT-1) att_qk_counter (.clk(clk),.rstn(rstn),.inc(att_qk_inc),.overflow(att_qk_cnt_overflow),.out(att_qk_cnt));
    // REUSE_NUM: Number of reuse loop 
    counter #(ATT_QK_LOOP-1) att_qk_loop_counter (.clk(clk),.rstn(rstn),.inc(att_qk_reuse_loop_inc),.overflow(att_qk_reuse_loop_overflow),.out(att_qk_reuse_loop_cnt));
    // ATT_PV_CNT: Number of activations in one single loop
    counter #(ATT_PV_CNT-1) att_pv_counter (.clk(clk),.rstn(rstn),.inc(att_pv_inc),.overflow(att_pv_cnt_overflow),.out(att_pv_cnt));
    // REUSE_NUM: Number of reuse loop 
    counter #(ATT_PV_LOOP-1) att_pv_loop_counter (.clk(clk),.rstn(rstn),.inc(att_pv_reuse_loop_inc),.overflow(att_pv_reuse_loop_overflow),.out(att_pv_reuse_loop_cnt));
    // ATTENTION SEQ Counter
    counter #(`SEQ_LENGTH-1) att_compute_seq_counter (.clk(clk),.rstn(rstn),.inc(att_compute_seq_inc),.overflow(att_compute_seq_overflow),.out(att_compute_seq_cnt));

endmodule

