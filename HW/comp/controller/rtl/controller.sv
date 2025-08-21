//FIXME:
//core_acc, controller, top, consmax
module controller #(
    parameter INST_REG_DEPTH = 4,              // Only for simulation, need to be reconsidered
    parameter PC_WIDTH = $clog2(INST_REG_DEPTH)
)(
    // Global Signals
    input                                                    clk,
    input                                                    rstn,
    // Global Config Signals
    //output  CFG_ARR_PACKET                                   arr_cfg,
    // Channel - Global Bus to Access Core Memory and MAC Result
    // 1. Write Channel
    //      1.1 Chip Interface -> WMEM for Weight Upload
    //      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
    //      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
    // 2. Read Channel
    //      2.1 WMEM       -> Chip Interface for Weight Check
    //      2.2 KV Cache   -> Chip Interface for KV Check
    //      2.3 MAC Result -> Vector Engine  for Post Processing
    output logic       [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-1:0]   gbus_addr,
    output CTRL                                              gbus_wen,
    output CTRL                                              gbus_ren,
    input  CTRL                                              gbus_rvalid,

    output logic                                             vlink_enable,
    // output logic       [`ARR_VNUM-1:0][`ARR_GBUS_DATA-1:0]   vlink_wdata,
    output logic       [`ARR_VNUM-1:0]                       vlink_wen,
    input              [`ARR_VNUM-1:0]                       vlink_rvalid,

//    output logic       [`ARR_HNUM-1:0][`ARR_GBUS_DATA-1:0]   hlink_wdata,    //hlink_wdata go through reg, to hlink_rdata
    output logic       [`ARR_HNUM-1:0]                       hlink_wen,
    input              [`ARR_HNUM-1:0]                       hlink_rvalid,

    //Global SRAM Access Bus
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]    global_sram_waddr,
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]    global_sram_raddr,
    output logic                                             global_sram_wen,
    output logic                                             global_sram_ren,
    output GSRAM_WSEL                                        global_sram_wsel,
    output GSRAM_RSEL                                        global_sram_rsel,

    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram0_waddr,
    output logic       [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]      global_sram0_raddr,
    output logic                                             global_sram0_wen,
    output logic                                             global_sram0_ren,
    output GSRAM_WSEL                                        global_sram0_wsel,
    output GSRAM_RSEL                                        global_sram0_rsel,

    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    output CMEM_ARR_PACKET                                   arr_cmem,
    // Local Buffer Access for Weight and KV Cache
    input  CTRL                                              lbuf_empty,
    input  CTRL                                              lbuf_reuse_empty,
    input  CTRL                                              lbuf_full,
    input  CTRL                                              lbuf_almost_full,
    output CTRL                                              lbuf_ren,
    output CTRL                                              lbuf_reuse_ren,
    output CTRL                                              lbuf_reuse_rst,
    // Local Buffer Access for Activation
    input  CTRL                                              abuf_empty,
    input  CTRL                                              abuf_reuse_empty,
    input  CTRL                                              abuf_full,
    input  CTRL                                              abuf_almost_full,
    output CTRL                                              abuf_reuse_ren,
    output CTRL                                              abuf_reuse_rst,
    output CTRL                                              abuf_ren,

    //Mux select signals
    output HLINK_WSEL                                        hlink_sel,
    output GBUS_WSEL                                         gbus_sel,
    output LN_WSEL                                           ln_sel,
    //vec engine valid
    output logic  [`ARR_HNUM-1:0]                            ctrl_cons_valid,
    output logic                                             ctrl_ln_valid,
    output logic  [`ARR_HNUM-1:0]                            ctrl_wb_valid,
    //SFR connection
    input                                                    start,
    input [$clog2(INST_REG_DEPTH)-1:0]                       inst_reg_addr,
    input                                                    inst_reg_wen,
    input                                                    inst_reg_ren,
    input  GPT_COMMAND                                       inst_reg_wdata,
    output GPT_COMMAND                                       inst_reg_rdata,

    input [`ARR_HNUM-1:0]                                    ctrl_cons_ovalid,
    output logic  [PC_WIDTH-1:0]                             pc_reg,

    output logic  [`ARR_HNUM-1:0][`ARR_CDATA_BIT-1:0]        cfg_acc_num,

    output logic clear_fifo_addr,

    input  logic wbfifo_empty2controller

);
/************* Top level FSM *************/
GPT_COMMAND [INST_REG_DEPTH-1:0] inst_reg;
always_ff @(posedge clk or negedge rstn) begin
    if(~rstn) begin
        inst_reg<='0;
        inst_reg_rdata<=0;
    end
    else if(inst_reg_wen) begin
        inst_reg[inst_reg_addr]<=inst_reg_wdata;
    end
    else if(inst_reg_ren) begin
        inst_reg_rdata<=inst_reg[inst_reg_addr];
    end
end
STATE state,next_state;

logic [PC_WIDTH-1:0] next_pc_reg;

//finishing flags for fsm
logic load_finish, compute_finish;

//Core address depth during load weight
parameter CORE_ADDR_CNT = `N_MODEL*`N_MODEL/`N_HEAD/`ARR_VNUM/(`ARR_GBUS_DATA/`ARR_IDATA_BIT); //384
// Total address depth for activation
parameter ABUF_CNT = `SEQ_LENGTH*`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT); //3840
// Total address depth for weight in each core
parameter LBUF_CNT = CORE_ADDR_CNT;
parameter LOAD_INCORE_WIDTH = $clog2(CORE_ADDR_CNT);
parameter BIT_WIDTH_H = $clog2(`ARR_HNUM);
parameter BIT_WIDTH_V = $clog2(`ARR_VNUM);

//load weight for QKV gen
//each core stalls n_model/n_head/n_head channel, each channel is n_model length.
//gbus_data is ARR_GBUS_DATA bits, each data is ARR_IDATA_BIT bits, a word line can save ARR_GBUS_DATA/ARR_IDATA_BIT number of data
//each core need n_model/n_head/n_head * n_model/(ARR_GBUS_DATA/ARR_IDATA_BIT) address
//first load the column (one head), then load the row (different heads)
//global sram read address accumulate by 1 each cycle. 

//====================================  Vgen signal definition =================================

/******VGEN loadweight gbus_addr generation******/
logic vload_finish, vcompute_finish;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0]   gbus_load_vgen_addr_tmp;     //gbus_load_vgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-1:0]   gbus_load_vgen_addr;         //gbus_load_vgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0]    global_sram_load_vgen_addr;  //extra 1 bit for differentiating writing to kv cache or wmem                                             
CTRL                                        gbus_load_vgen_wen;

/************* LOAD_WEIGHT counter *************/
logic vload_core_inc;
logic vload_core_inc_d;
logic vload_in_core_overflow;
logic [LOAD_INCORE_WIDTH-1:0] vload_in_core_cnt;

logic vload_core_col_overflow;
logic [BIT_WIDTH_V-1:0] vload_core_col_cnt;

logic vload_core_row_overflow;
logic [BIT_WIDTH_H-1:0] vload_core_row_cnt;

/************* VGEN Compute Registers *************/
parameter WEI_VGEN_NUM = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT)); //384 / ( 64 / 8 ) = 48
parameter WEI_REUSE_VGEN_NUM = `SEQ_LENGTH;
parameter ACT_VGEN_NUM = (`N_MODEL*`SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT)); //64*384/8=3072

parameter WB_VGEN_NUM = (`SEQ_LENGTH*`N_MODEL/`N_HEAD/`ARR_VNUM/(`ARR_GBUS_DATA/`ARR_IDATA_BIT)); //64*384/6/8/8=64
parameter ACT_VGEN_NUM_BITS = $clog2(ACT_VGEN_NUM);
parameter WEI_REUSE_VGEN_NUM_BITS = $clog2(WEI_REUSE_VGEN_NUM);

parameter WEI_VGEN_NUM_BITS=$clog2(WEI_VGEN_NUM);
parameter WB_VGEN_NUM_BITS=$clog2(WB_VGEN_NUM);

parameter ABUF_VGEN_LOAD_NUM = (`N_MODEL/`N_HEAD/`ARR_VNUM); //8
parameter ABUF_VGEN_LOAD_NUM_BITS = $clog2(ABUF_VGEN_LOAD_NUM);

CTRL                                                        cmem_compute_vgen_ren;
G_ADDR                                                      cmem_compute_vgen_raddr;
logic [`ARR_GBUS_ADDR-2:0]                                  cmem_compute_vgen_raddr_tmp;
CTRL                                                        cmem_compute_vgen_wen;
logic [`ARR_GBUS_ADDR-2:0]                                  cmem_wb_compute_vgen_waddr_tmp;
G_ADDR                                                      cmem_wb_compute_vgen_waddr;

CTRL                                                        abuf_compute_vgen_ren;
CTRL                                                        abuf_compute_vgen_reuse_ren;
CTRL                                                        abuf_compute_vgen_reuse_rst;

CTRL                                                        lbuf_compute_vgen_ren;
CTRL                                                        lbuf_compute_vgen_reuse_ren;
CTRL                                                        lbuf_compute_vgen_reuse_rst;

logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                      global_sram_compute_vgen_addr;
logic                                                       global_sram_compute_vgen_ren;

logic [`ARR_HNUM-1:0]                                       hlink_compute_vgen_wen;

logic                                                       vcompute_act_overflow;
logic                                                       vcompute_act_inc;
logic [ACT_VGEN_NUM_BITS-1:0]                               vcompute_act_cnt;
logic [WEI_REUSE_VGEN_NUM_BITS-1:0]                         vcompute_wei_reuse_cnt;
logic                                                       vcompute_wei_reuse_overflow;

logic                                                       vcompute_wei_overflow;
logic [WEI_VGEN_NUM_BITS-1:0]                               vcompute_wei_cnt;

logic                                                       vcompute_wb_col_inc;
logic                                                       vcompute_wb_inc;
logic                                                       vcompute_wb_overflow;
logic                                                       vcompute_wb_col_overflow;
logic [WB_VGEN_NUM_BITS-1:0]                                vcompute_wb_cnt;
logic [BIT_WIDTH_V-1:0]                                      vcompute_wb_col_cnt;

logic [ABUF_VGEN_LOAD_NUM_BITS:0]                         abuf_vcompute_load_num;

CTRL                                                        gbus_compute_vgen_ren;


//==================================== Kgen signal definition =================================
/******KGEN loadweight gbus_addr generation******/
logic kload_finish, kcompute_finish;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0] gbus_load_kgen_addr_tmp; //gbus_load_kgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-1:0] gbus_load_kgen_addr; //gbus_load_kgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0] global_sram_load_kgen_addr;//extra 1 bit for differentiating writing to kv cache or wmem                                             
CTRL                       gbus_load_kgen_wen;

/************* LOAD_WEIGHT counter *************/
logic kload_core_inc;
logic kload_core_inc_d;
logic kload_in_core_overflow;
logic [LOAD_INCORE_WIDTH-1:0] kload_in_core_cnt;

logic kload_core_col_overflow;
logic [BIT_WIDTH_V-1:0] kload_core_col_cnt;

logic kload_core_row_overflow;
logic [BIT_WIDTH_H-1:0] kload_core_row_cnt;

/************* KGEN Compute Registers *************/
parameter ACT_KGEN_NUM = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT));                   // 384 / ( 64 / 8 ) = 48
parameter ACT_REUSE_KGEN_NUM = (`N_MODEL/`N_HEAD/`ARR_VNUM);
parameter ACT_KGEN_NUM_BITS=$clog2(ACT_KGEN_NUM);
parameter ACT_REUSE_KGEN_NUM_BITS=$clog2(ACT_REUSE_KGEN_NUM);

parameter WEI_KGEN_NUM = (ACT_KGEN_NUM*`N_MODEL/`N_HEAD/`ARR_VNUM); //1 row Activation reuse addr cnt
parameter WB_KCOL_CNT = (`SEQ_LENGTH/`ARR_VNUM*`N_MODEL/`N_HEAD/ (`ARR_GBUS_DATA/`ARR_IDATA_BIT)); //80/8*384/6/(64/8) = 80
parameter WEI_KGEN_NUM_BITS=$clog2(WEI_KGEN_NUM);

parameter LBUF_KGEN_LOAD_NUM = `SEQ_LENGTH;
parameter LBUF_KGEN_LOAD_NUM_BITS = $clog2(LBUF_KGEN_LOAD_NUM);

CTRL                                                        cmem_compute_kgen_ren;
G_ADDR                                                      cmem_compute_kgen_raddr;
logic [`ARR_GBUS_ADDR-2:0]                                  cmem_compute_kgen_raddr_tmp;

CTRL                                                        abuf_compute_kgen_ren;
CTRL                                                        abuf_compute_kgen_reuse_ren;
CTRL                                                        abuf_compute_kgen_reuse_rst;

CTRL                                                        lbuf_compute_kgen_ren;
CTRL                                                        lbuf_compute_kgen_reuse_ren;
CTRL                                                        lbuf_compute_kgen_reuse_rst;


logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                      global_sram_compute_kgen_addr;
logic                                                       global_sram_compute_kgen_ren;

logic [`ARR_HNUM-1:0]                                       hlink_compute_kgen_wen;

logic                                                       kcompute_act_overflow;
logic [ACT_KGEN_NUM_BITS-1:0]                               kcompute_act_cnt;
logic [ACT_REUSE_KGEN_NUM_BITS-1:0]                         kcompute_act_reuse_cnt;
logic                                                       kcompute_act_reuse_overflow;

logic                                                       kcompute_wei_inc;
logic                                                       kcompute_wei_overflow;
logic [WEI_KGEN_NUM_BITS-1:0]                               kcompute_wei_cnt;

logic                                                       kcompute_wb_col_inc;
logic                                                       kcompute_wb_inc;
logic                                                       kcmpute_wb_times_inc;
logic                                                       kcompute_wb_overflow;
logic                                                       kcompute_wb_col_overflow;
logic [$clog2(WB_KCOL_CNT)-1:0]                             kcompute_wb_cnt;
logic [BIT_WIDTH_V-1:0]                                     kcompute_wb_col_cnt;

logic [LBUF_KGEN_LOAD_NUM_BITS:0]                         lbuf_kcompute_load_num;

CTRL                                                        gbus_compute_kgen_ren;
CTRL                                                        gbus_compute_kgen_wen;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-1:0]                   gbus_compute_kgen_addr;
logic [`ARR_GBUS_ADDR-2:0]                                  gbus_compute_kgen_addr_tmp;



//==================================== Qgen signal definition =================================

/******QGEN loadweight gbus_addr generation******/
logic qload_finish, qcompute_finish;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0] gbus_load_qgen_addr_tmp;
logic [`ARR_HNUM-1:0][`ARR_GBUS_ADDR-2:0] gbus_load_qgen_addr; //gbus_load_qgen_addr[`ARR_GBUS_ADDR-1] should always be 0 in this circumstances
logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0] global_sram_load_qgen_addr;//extra 1 bit for differentiating writing to kv cache or wmem                                             
CTRL                       gbus_load_qgen_wen;

/************* LOAD_WEIGHT counter *************/
logic qload_core_inc;
logic qload_core_inc_d;
logic qload_in_core_overflow;
logic [LOAD_INCORE_WIDTH-1:0] qload_in_core_cnt; //was BIT_WIDTH-1

logic qload_core_col_overflow;
logic [BIT_WIDTH_V-1:0] qload_core_col_cnt; //was BIT_WIDTH-1

logic qload_core_row_overflow;
logic [BIT_WIDTH_H-1:0] qload_core_row_cnt; //was BIT_WIDTH-1


/************* QGEN Compute Registers *************/
parameter ACT_QGEN_NUM = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT));
parameter ACT_REUSE_QGEN_NUM = (`N_MODEL/`N_HEAD/`ARR_VNUM);
parameter ACT_QGEN_NUM_BITS=$clog2(ACT_QGEN_NUM);
parameter ACT_REUSE_QGEN_NUM_BITS=$clog2(ACT_REUSE_QGEN_NUM);
parameter WEI_QGEN_NUM = (ACT_KGEN_NUM*`N_MODEL/`N_HEAD/`ARR_VNUM);

parameter WB_QGEN_NUM = (`SEQ_LENGTH*`N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT)); //ADDR space needed to write back SRAM

parameter WEI_QGEN_NUM_BITS=$clog2(WEI_QGEN_NUM);
parameter WB_QGEN_NUM_BITS=$clog2(WB_QGEN_NUM);

parameter LBUF_QGEN_LOAD_NUM = `SEQ_LENGTH;
parameter LBUF_QGEN_LOAD_NUM_BITS = $clog2(LBUF_QGEN_LOAD_NUM);


CTRL                                                        cmem_compute_qgen_ren;
G_ADDR                                                      cmem_compute_qgen_raddr;
logic [`ARR_GBUS_ADDR-2:0]                                  cmem_compute_qgen_raddr_tmp;

CTRL                                                        abuf_compute_qgen_ren;
CTRL                                                        abuf_compute_qgen_reuse_ren;
CTRL                                                        abuf_compute_qgen_reuse_rst;

CTRL                                                        lbuf_compute_qgen_ren;
CTRL                                                        lbuf_compute_qgen_reuse_ren;
CTRL                                                        lbuf_compute_qgen_reuse_rst;


logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]                      global_sram_compute_qgen_addr;
logic                                                       global_sram_compute_qgen_ren;

logic [$clog2(`GLOBAL_SRAM_DEPTH+1)-1:0]                    global_sram_compute_qgen_waddr;
logic                                                       global_sram_compute_qgen_wen;


logic [`ARR_HNUM-1:0]                                       hlink_compute_qgen_wen;

logic [`ARR_VNUM-1:0]                                       labuf_compute_qgen_ren;

logic                                                       qcompute_act_overflow;
logic [ACT_QGEN_NUM_BITS-1:0]                               qcompute_act_cnt;
logic [ACT_REUSE_QGEN_NUM_BITS-1:0]                         qcompute_act_reuse_cnt;
logic                                                       qcompute_act_reuse_overflow;

logic                                                       qcompute_wei_inc;
logic                                                       qcompute_wei_overflow;
logic [WEI_QGEN_NUM_BITS-1:0]                               qcompute_wei_cnt;

CTRL                                                        cmem_compute_qgen_wen;

logic [LBUF_QGEN_LOAD_NUM_BITS:0] lbuf_qcompute_load_num;


CTRL                                                        qcompute_gbus_ren;

//====================================  Attention signal definition =================================
parameter GSRAM_ATT_QADDR_CNT = `N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
parameter VEC_ATT_SADDR_CNT = `SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
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
//attention cfg acc num
logic  [`ARR_HNUM-1:0][`ARR_CDATA_BIT-1:0]        att_cfg_acc_num;
/******Attention Results Writeback******/
parameter ATT_OUT_WB = `N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
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
logic [`ARR_HNUM-1:0]   att_ctrl_wb_valid;
logic [`ARR_HNUM-2:0]   att_ctrl_wb_valid_reg;
logic att_ctrl_wb_valid_w;
logic [`ARR_HNUM-1:0]   att_ctrl_cons_valid;
logic [`ARR_HNUM-2:0]   att_ctrl_cons_valid_reg;
logic att_ctrl_cons_valid_w;
logic att_wb_finish, next_att_wb_finish;
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
parameter ATT_QK_CNT = `N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
//Max count for loop
parameter ATT_QK_LOOP = `SEQ_LENGTH/(`ARR_VNUM);
logic                                      att_qk_cnt_overflow;
logic [$clog2(ATT_QK_CNT)-1:0]             att_qk_cnt;
logic                                      att_qk_inc;
logic                                      att_qk_reuse_loop_overflow;
logic [$clog2(ATT_QK_LOOP)-1:0]            att_qk_reuse_loop_cnt;
logic                                      att_qk_reuse_loop_inc;
//Attention PV reuse counter
//Max count for each Loop for P reuse during PV
parameter ATT_PV_CNT = `SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT);
//Max count for loop
parameter ATT_PV_LOOP = `N_MODEL/`N_HEAD/(`ARR_VNUM);
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

    //////////////////////////////////////////////////
    //                                              //
    //       FSM                                    //
    //                                              //
    //////////////////////////////////////////////////
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        state<=IDLE;
        pc_reg <= 0;
    end
    else begin
        state<=next_state;
        pc_reg <= next_pc_reg;
    end
end

always_comb begin
    next_state = state;
    next_pc_reg = pc_reg;
    case(state)
        IDLE: begin
            next_state = start ? LOAD_WEIGHT : IDLE;
            next_pc_reg = pc_reg;
        end
        LOAD_WEIGHT: begin
            next_state = load_finish ? COMPUTE : LOAD_WEIGHT;
            next_pc_reg = pc_reg;
        end
        COMPUTE: begin
            next_state = compute_finish ? FINISH : COMPUTE;
            next_pc_reg = pc_reg;
        end
        FINISH: begin
            if(pc_reg != INST_REG_DEPTH-1) begin
                next_state = LOAD_WEIGHT;
                next_pc_reg = pc_reg + 1;
            end
            else begin
                next_state = COMPLETE;
                next_pc_reg = pc_reg;
            end
        end
        COMPLETE: begin
            next_state = COMPLETE;
            next_pc_reg = pc_reg;
        end
    endcase
end


    //////////////////////////////////////////////////
    //                                              //
    //       Global Control Signal Select           //
    //                                              //
    //////////////////////////////////////////////////
//gbus_addr, gbus_wen
always_comb begin
    clear_fifo_addr = 1'b0;
    vlink_enable    = 1'b0;
    vlink_wen       = '0;

    gbus_addr = '0;
    gbus_wen = '0;
    gbus_ren = '0;

    arr_cmem.cmem_raddr = '0;
    arr_cmem.cmem_ren = '0;
    arr_cmem.cmem_wen = '0;
    arr_cmem.cmem_waddr = '0;

    hlink_wen = '0;

    global_sram_raddr = '0;
    global_sram_ren   = '0;
    global_sram_rsel = GSRAM2SPI;

    global_sram_wen = '0;
    global_sram_waddr = '0;
    global_sram_wsel = SPI2GSRAM;

    global_sram0_ren = '0;
    global_sram0_raddr = '0;
    global_sram0_rsel = GSRAM2SPI;
    global_sram0_waddr = '0;
    global_sram0_wen = '0;
    global_sram0_wsel = SPI2GSRAM;

    lbuf_ren = '0;
    lbuf_reuse_ren = '0;
    lbuf_reuse_rst = '0;

    abuf_ren = '0;
    abuf_reuse_ren = '0;
    abuf_reuse_rst = '0;

    // May need to reconsider
    for(int i=0;i<`ARR_HNUM;i++) begin
        cfg_acc_num[i] = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT) - 1);         // 384 / ( 64 / 8 ) - 1 = 47
    end

    hlink_sel = GSRAM02HLINK;
    gbus_sel = SPI2GBUS;
    ln_sel = GSRAM02LN;

    ctrl_cons_valid = 'b0;
    ctrl_ln_valid = 1'b0;
    ctrl_wb_valid = 'b0;

    load_finish = 1'b0;
    compute_finish = 1'b0;

    // /*Only for top level debug*/
    // inst_reg[0] = Q_GEN;
    // inst_reg[1] = K_GEN;
    // inst_reg[2] = V_GEN;
    // inst_reg[3] = ATT;
    // /*Only for top level debug*/
    case(inst_reg[pc_reg]) 
        NOP: begin
            load_finish = 1'b1;
            compute_finish = 1'b1;
        end
        Q_GEN: begin
            gbus_addr = (state == LOAD_WEIGHT)          ?   gbus_load_qgen_addr             : '0;
            gbus_wen = (state == LOAD_WEIGHT)           ?   gbus_load_qgen_wen              : '0;
            gbus_ren = '0;

            arr_cmem.cmem_raddr = (state == COMPUTE)    ?   cmem_compute_qgen_raddr         : '0;
            arr_cmem.cmem_ren = (state == COMPUTE)      ?   cmem_compute_qgen_ren           : '0;
            hlink_wen = (state == COMPUTE)              ?   hlink_compute_qgen_wen          : '0;

            global_sram_raddr = (state == LOAD_WEIGHT)  ?   global_sram_load_qgen_addr      :
                                (state == COMPUTE)      ?   global_sram_compute_qgen_addr   :
                                                            '0;
            global_sram_ren   = (state == LOAD_WEIGHT)  ?   qload_core_inc                  : 
                                (state == COMPUTE)      ?   global_sram_compute_qgen_ren    :
                                                            '0;
            global_sram_rsel = (state == COMPUTE)       ?   GSRAM2CHIP : GSRAM2SPI;
            global_sram_wen = '0;
            global_sram_waddr = '0;  
            global_sram_wsel = (state == COMPUTE) ? WSEL_DISABLE : SPI2GSRAM;

            global_sram0_ren = '0;
            global_sram0_raddr = '0;
            global_sram0_rsel = (state == COMPUTE)       ?   RSEL_DISABLE : GSRAM2SPI;
            global_sram0_waddr = (state == COMPUTE)      ?   global_sram_compute_qgen_waddr  :
                                                            '0;
            global_sram0_wen = (state == COMPUTE)        ?   global_sram_compute_qgen_wen    : '0;
            global_sram0_wsel =(state == COMPUTE)        ?   FIFO2GSRAM : SPI2GSRAM;

            lbuf_ren = (state == COMPUTE)               ?   lbuf_compute_qgen_ren           : '0;
            lbuf_reuse_ren = (state == COMPUTE)         ?   lbuf_compute_qgen_reuse_ren     : '0;
            lbuf_reuse_rst = (state == COMPUTE)         ?   lbuf_compute_qgen_reuse_rst     : '0;
            abuf_ren = (state == COMPUTE)               ?   abuf_compute_qgen_ren           : '0;
            abuf_reuse_ren = (state == COMPUTE)         ?   abuf_compute_qgen_reuse_ren     : '0;
            abuf_reuse_rst = (state == COMPUTE)         ?   abuf_compute_qgen_reuse_rst     : '0;

            for(int i=0;i<`ARR_HNUM;i++) begin
                cfg_acc_num[i] = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT) - 1);         // 384 / ( 64 / 8 ) - 1 = 47
            end
            hlink_sel = GSRAM12HLINK;
            gbus_sel = (state == COMPUTE)              ?    GBUS2GBUS                       : SPI2GBUS;
            ln_sel = GSRAM02LN;

            ctrl_cons_valid = 1'b0;
            ctrl_ln_valid = 1'b0;
            for(int i=0;i<`ARR_HNUM;i++) begin
                ctrl_wb_valid[i] = (state == COMPUTE) ? global_sram0_wen : 1'b0;
            end

            load_finish = qload_finish;
            compute_finish = qcompute_finish;
            clear_fifo_addr = (state == FINISH);

        end

        V_GEN: begin
            gbus_addr = (state == LOAD_WEIGHT)          ?   gbus_load_vgen_addr             : '0;
            gbus_wen = (state==LOAD_WEIGHT)             ?   gbus_load_vgen_wen              : '0;
            gbus_ren = (state==COMPUTE)                 ?   gbus_compute_vgen_ren           : '0;

            arr_cmem.cmem_raddr = (state == COMPUTE)    ?   cmem_compute_vgen_raddr         : '0;
            arr_cmem.cmem_ren = (state == COMPUTE)      ?   cmem_compute_vgen_ren           : '0;
            arr_cmem.cmem_wen = (state == COMPUTE)      ?   cmem_compute_vgen_wen           : '0;
            arr_cmem.cmem_waddr = (state == COMPUTE)    ?   cmem_wb_compute_vgen_waddr       : '0;

            hlink_wen = (state == COMPUTE)              ?   hlink_compute_vgen_wen          : '0;

            global_sram_raddr = (state == LOAD_WEIGHT)  ?   global_sram_load_vgen_addr      :
                                (state == COMPUTE)      ?   global_sram_compute_vgen_addr   :
                                                            '0;
            global_sram_ren   = (state == LOAD_WEIGHT)  ?   vload_core_inc                  : 
                                (state == COMPUTE)      ?   global_sram_compute_vgen_ren    :
                                                            '0;
            global_sram_rsel = (state == COMPUTE)       ?   GSRAM2CHIP : GSRAM2SPI;

            global_sram_wen = '0;
            global_sram_waddr = '0;
            global_sram_wsel = (state == COMPUTE) ? WSEL_DISABLE : SPI2GSRAM;

            global_sram0_ren = '0;
            global_sram0_raddr = '0;
            global_sram0_rsel = (state == COMPUTE)       ?   RSEL_DISABLE : GSRAM2SPI;
            global_sram0_waddr = '0;
            global_sram0_wen = '0;
            global_sram0_wsel = (state == COMPUTE) ? WSEL_DISABLE : SPI2GSRAM;


            lbuf_ren = (state == COMPUTE)               ?   lbuf_compute_vgen_ren           : '0;
            lbuf_reuse_ren = (state == COMPUTE)         ?   lbuf_compute_vgen_reuse_ren     : '0;
            lbuf_reuse_rst = (state == COMPUTE)         ?   lbuf_compute_vgen_reuse_rst     : '0;

            abuf_ren = (state == COMPUTE)               ?   abuf_compute_vgen_ren           : '0;
            abuf_reuse_ren = (state == COMPUTE)         ?   abuf_compute_vgen_reuse_ren     : '0;
            abuf_reuse_rst = (state == COMPUTE)         ?   abuf_compute_vgen_reuse_rst     : '0;

            for(int i=0;i<`ARR_HNUM;i++) begin
                cfg_acc_num[i] = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT) - 1);         // 384 / ( 64 / 8 ) - 1 = 47
            end

            hlink_sel = GSRAM12HLINK;
            gbus_sel = (state == COMPUTE)    ?    GBUS2GBUS                      : SPI2GBUS;
            ln_sel = GSRAM02LN;

            ctrl_cons_valid = 'b0;
            ctrl_ln_valid = 1'b0;
            ctrl_wb_valid = 'b0;

            load_finish = vload_finish;
            compute_finish = vcompute_finish;
            clear_fifo_addr = (state == FINISH);

        end

        K_GEN: begin
            gbus_addr = (state == LOAD_WEIGHT)          ?   gbus_load_kgen_addr             :
                        (state == COMPUTE)              ?   gbus_compute_kgen_addr          : '0;
            gbus_wen = (state==LOAD_WEIGHT)             ?   gbus_load_kgen_wen              :
                       (state == COMPUTE)               ?   gbus_compute_kgen_wen           : '0;
            gbus_ren = (state==COMPUTE)                 ?   gbus_compute_kgen_ren           : '0;

            arr_cmem.cmem_raddr = (state == COMPUTE)    ?   cmem_compute_kgen_raddr         : '0;
            arr_cmem.cmem_ren = (state == COMPUTE)      ?   cmem_compute_kgen_ren           : '0;
            arr_cmem.cmem_wen = '0;
            arr_cmem.cmem_waddr = '0;

            hlink_wen = (state == COMPUTE)              ?   hlink_compute_kgen_wen          : '0;

            global_sram_raddr = (state == LOAD_WEIGHT)  ?   global_sram_load_kgen_addr      :
                                (state == COMPUTE)      ?   global_sram_compute_kgen_addr   :
                                                            '0;
            global_sram_ren   = (state == LOAD_WEIGHT)  ?   kload_core_inc                  : 
                                (state == COMPUTE)      ?   global_sram_compute_kgen_ren    :
                                                            '0;
            global_sram_rsel = (state == COMPUTE)       ?   GSRAM2CHIP : GSRAM2SPI;

            global_sram_wen = '0;
            global_sram_waddr = '0;
            global_sram_wsel = (state == COMPUTE) ? WSEL_DISABLE : SPI2GSRAM;

            global_sram0_ren = '0;
            global_sram0_raddr = '0;
            global_sram0_rsel = (state == COMPUTE)       ?   RSEL_DISABLE : GSRAM2SPI;
            global_sram0_waddr = '0;
            global_sram0_wen = '0;
            global_sram0_wsel = (state == COMPUTE) ? WSEL_DISABLE : SPI2GSRAM;

            lbuf_ren = (state == COMPUTE)               ?   lbuf_compute_kgen_ren           : '0;
            lbuf_reuse_ren = (state == COMPUTE)         ?   lbuf_compute_kgen_reuse_ren     : '0;
            lbuf_reuse_rst = (state == COMPUTE)         ?   lbuf_compute_kgen_reuse_rst     : '0;

            abuf_ren = (state == COMPUTE)               ?   abuf_compute_kgen_ren           : '0;
            abuf_reuse_ren = (state == COMPUTE)         ?   abuf_compute_kgen_reuse_ren     : '0;
            abuf_reuse_rst = (state == COMPUTE)         ?   abuf_compute_kgen_reuse_rst     : '0;

            for(int i=0;i<`ARR_HNUM;i++) begin
                cfg_acc_num[i] = (`N_MODEL/(`ARR_GBUS_DATA/`ARR_IDATA_BIT) - 1);         // 384 / ( 64 / 8 ) - 1 = 47
            end

            hlink_sel = GSRAM12HLINK;
            gbus_sel = (state == COMPUTE)              ?    GBUS2GBUS                       : SPI2GBUS;
            ln_sel = GSRAM02LN;

            ctrl_cons_valid = 'b0;
            ctrl_ln_valid = 1'b0;
            ctrl_wb_valid = 'b0;

            load_finish = kload_finish;
            compute_finish = kcompute_finish;
            clear_fifo_addr = (state == FINISH);

        end

        ATT: begin
            gbus_addr = '0;
            gbus_wen  = '0;
            gbus_ren  = '0;

            load_finish = 1'b1;
            compute_finish = (state == COMPUTE )        ?   att_wb_finish                   : '0;
            arr_cmem       = (state == COMPUTE )        ?   att_cmem                        : '0;
            hlink_wen      = (state == COMPUTE)         ?   hlink_att_wen                   : '0;

            global_sram0_raddr = (state == COMPUTE)      ?   global_sram_att_q_raddr         : '0;
            global_sram0_ren   = (state == COMPUTE)      ?   global_sram_att_q_ren           : '0;
            global_sram0_rsel = (state == COMPUTE)   ?   global_sram_att_q_rsel          : GSRAM2SPI;
            global_sram0_waddr = '0;
            global_sram0_wen = '0;
            global_sram0_wsel = (state == COMPUTE) ? WSEL_DISABLE : SPI2GSRAM;

            global_sram_raddr = '0;
            global_sram_ren   = '0;
            global_sram_rsel = (state == COMPUTE)       ?   RSEL_DISABLE : GSRAM2SPI;
            global_sram_waddr = '0;
            global_sram_wen = '0;
            global_sram_wsel = (state == COMPUTE) ? FIFO2GSRAM : SPI2GSRAM;

            hlink_sel = att_hlink_sel;
            gbus_sel =  (state == COMPUTE) ? GBUS2GBUS : SPI2GBUS;
            ln_sel =    GSRAM02LN;
            
            lbuf_ren = (state == COMPUTE)               ?   (att_k_lbuf_ren | att_v_lbuf_ren)           : '0;
            lbuf_reuse_ren = (state == COMPUTE)         ?   (att_k_lbuf_reuse_ren | att_v_lbuf_reuse_ren)     : '0;
            lbuf_reuse_rst = (state == COMPUTE)         ?   (att_k_lbuf_reuse_rst | att_v_lbuf_reuse_rst)     : '0;

            abuf_ren = (state == COMPUTE)               ?   (att_q_abuf_ren | att_p_abuf_ren)            : '0;
            abuf_reuse_ren = (state == COMPUTE)         ?   (att_q_abuf_reuse_ren | att_p_abuf_reuse_ren): '0;
            abuf_reuse_rst = (state == COMPUTE)         ?   (att_q_abuf_reuse_rst | att_p_abuf_reuse_rst): '0;

            cfg_acc_num = att_cfg_acc_num;

            ctrl_cons_valid = att_ctrl_cons_valid;
            ctrl_ln_valid = 1'b0;
            ctrl_wb_valid = att_ctrl_wb_valid;
            clear_fifo_addr = (state == FINISH);
        end

    endcase
end

    //////////////////////////////////////////////////
    //                                              //
    //                VGEN                          //
    //                                              //
    //////////////////////////////////////////////////

genvar i,j;
assign vload_core_inc=(inst_reg[pc_reg] == V_GEN) && (state == LOAD_WEIGHT) && !vload_core_row_overflow;
//load_finish signal
assign vload_finish = 1;

//VGEN inst, load weight state, gbus_addr logic
always_comb begin
    for(int i=0;i<`ARR_HNUM;i++) begin
        if(i==vload_core_row_cnt)
            gbus_load_vgen_addr_tmp[i]=vload_in_core_cnt;
        else
            gbus_load_vgen_addr_tmp[i]='0;
    end
end

generate
    for(i=0;i<`ARR_HNUM;i++) begin
        assign gbus_load_vgen_addr[i] = {1'b0, gbus_load_vgen_addr_tmp[i]};
    end
endgenerate

//VGEN inst, load weight state, gbus_wen logic
generate
    for(i=0;i<`ARR_HNUM;i++) begin
        for(j=0;j<`ARR_VNUM;j++) begin
            always_comb begin
                if(vload_core_row_cnt==i && vload_core_col_cnt==j)
                    gbus_load_vgen_wen[i][j] = 1'b1;
                else
                    gbus_load_vgen_wen[i][j] = 1'b0;
            end
        end
    end
endgenerate

//VGEN inst, load weight state, global sram read addr
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        global_sram_load_vgen_addr <= `GLOBAL_SRAM_LOAD_VGEN_BASE_ADDR;
    end
    else if(vload_core_inc) begin //global sram raddr for vgen load weight
        global_sram_load_vgen_addr <= global_sram_load_vgen_addr + 1;
    end
end 

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        vload_core_inc_d<='0;
    end
    else begin
        vload_core_inc_d <= vload_core_inc;
    end
end


/******VGEN Compute address generation******/
logic wei_vgen_reuse_finish;
logic lbuf_vcompute_raddr_inc;
logic vcompute_core_en;

logic abuf_vcompute_raddr_inc;
assign vcompute_core_en = (inst_reg[pc_reg]==V_GEN) && (state==COMPUTE);
assign abuf_vcompute_raddr_inc = vcompute_core_en && (!abuf_almost_full[0][0]) && ((abuf_vcompute_load_num < (ABUF_VGEN_LOAD_NUM-1)) || (abuf_vcompute_load_num == (ABUF_VGEN_LOAD_NUM-1)) && (vcompute_act_cnt < ACT_VGEN_NUM));
assign lbuf_vcompute_raddr_inc = vcompute_core_en && (!lbuf_almost_full[0][0]) && (cmem_compute_vgen_raddr_tmp < (LBUF_CNT-1));

assign vcompute_finish = vcompute_wb_overflow;

assign wei_vgen_reuse_finish = (vcompute_wei_reuse_cnt == WEI_REUSE_VGEN_NUM-1);


// Each column shares the same cmem read address & cmem read enable
generate
    for(i=0;i<`ARR_HNUM;i++) begin
        assign cmem_compute_vgen_raddr[i][0] = {1'b0, cmem_compute_vgen_raddr_tmp};
        for(j=0;j<`ARR_VNUM;j++) begin
            assign cmem_wb_compute_vgen_waddr[i][j] = {1'b1, cmem_wb_compute_vgen_waddr_tmp};
        end
    end
endgenerate

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            for(int j=1;j<`ARR_VNUM;j++) begin
                cmem_compute_vgen_raddr[i][j] <= 0;
            end
        end
    end
    else begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            for(int j=1;j<`ARR_VNUM;j++) begin
                cmem_compute_vgen_raddr[i][j] <= cmem_compute_vgen_raddr[i][j-1];
            end
        end
    end
end

// cmem read enable

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        cmem_compute_vgen_ren <= '0;
    end
    else begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            for(int j=0;j<`ARR_VNUM;j++) begin
                if(j==0) begin
                    cmem_compute_vgen_ren[i][j] <= lbuf_vcompute_raddr_inc;
                end
                else begin
                    cmem_compute_vgen_ren[i][j] <= cmem_compute_vgen_ren[i][j-1];
                end
            end
        end
    end
end

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        cmem_compute_vgen_raddr_tmp <= '0;
    end
    else if(vcompute_core_en) begin
        if(cmem_compute_vgen_ren[0][0])
            cmem_compute_vgen_raddr_tmp <= cmem_compute_vgen_raddr_tmp + 1;
        else
            cmem_compute_vgen_raddr_tmp <= cmem_compute_vgen_raddr_tmp;
    end
    else
        cmem_compute_vgen_raddr_tmp <= '0;
end 

assign global_sram_compute_vgen_addr = vcompute_act_cnt;
assign global_sram_compute_vgen_ren = abuf_vcompute_raddr_inc;

// Hlink wen in the next cycle of MAC & quantization of the first column
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn)
        hlink_compute_vgen_wen <= '0;
    else begin
        for(int i=0; i<`ARR_HNUM; i++)
            hlink_compute_vgen_wen[i] <= abuf_vcompute_raddr_inc;
    end
end

// MAC & quantization, if kcompute_act_reuse_cnt == `ACT_REUSE_NUM - 1 (act_kgen_reuse_finish == 1), then ren, otherwise ruse_en
generate
    for(i=0; i<`ARR_HNUM; i++) begin
        assign lbuf_compute_vgen_ren[i][0] = wei_vgen_reuse_finish && (!abuf_empty[i][0]) && (!lbuf_empty[i][0]) ? 1 : 0;
        assign lbuf_compute_vgen_reuse_ren[i][0] = (!wei_vgen_reuse_finish) && (!abuf_empty[i][0]) && (!lbuf_reuse_empty[i][0]) ? 1 : 0;
        assign lbuf_compute_vgen_reuse_rst[i][0] = ((vcompute_wei_cnt == (WEI_VGEN_NUM-1)) && lbuf_compute_vgen_reuse_ren[i][0]) ? 1 : 0;
        assign abuf_compute_vgen_ren[i][0] = lbuf_compute_vgen_ren[i][0] || lbuf_compute_vgen_reuse_ren[i][0];
    end
endgenerate


assign abuf_compute_vgen_reuse_ren = '0;
assign abuf_compute_vgen_reuse_rst = '0;

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                abuf_compute_vgen_ren[j][i] <= 0;
                lbuf_compute_vgen_ren[j][i] <= 0;
                lbuf_compute_vgen_reuse_ren[j][i] <= 0;
                lbuf_compute_vgen_reuse_rst[j][i] <= 0;
            end
        end
    end
    else begin
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                abuf_compute_vgen_ren[j][i] <= abuf_compute_vgen_ren[j][i-1];
                lbuf_compute_vgen_ren[j][i] <= lbuf_compute_vgen_ren[j][i-1];
                lbuf_compute_vgen_reuse_ren[j][i] <= lbuf_compute_vgen_reuse_ren[j][i-1];
                lbuf_compute_vgen_reuse_rst[j][i] <= lbuf_compute_vgen_reuse_rst[j][i-1];
            end
        end
    end
end 

// Write back to V cache

// Decide which col to write back
always_comb begin
    for(int i=0; i<`ARR_HNUM; i++) begin
        for(int j=0; j<`ARR_VNUM; j++) begin
            if((j == vcompute_wb_col_cnt) && (|gbus_rvalid)) begin
                cmem_compute_vgen_wen[i][j] = 1;
            end
            else cmem_compute_vgen_wen[i][j] = 0;
        end
    end
end 

// Writing back address

assign cmem_wb_compute_vgen_waddr_tmp = vcompute_wb_cnt + `CMEM_VBASE_ADDR;


// Read channel is not useful here
assign gbus_compute_vgen_ren = '0;


    //////////////////////////////////////////////////
    //                                              //
    //                kGEN                          //
    //                                              //
    //////////////////////////////////////////////////

assign kload_core_inc=(inst_reg[pc_reg] == K_GEN) && (state == LOAD_WEIGHT) && !kload_core_row_overflow;
//load_finish signal
assign kload_finish = 1;

//KGEN inst, load weight state, gbus_addr logic
always_comb begin
    for(int i=0;i<`ARR_HNUM;i++) begin
        if(i == kload_core_row_cnt)
            gbus_load_kgen_addr_tmp[i]=kload_in_core_cnt;
        else
            gbus_load_kgen_addr_tmp[i]='0;
    end
end

generate
    for(i=0;i<`ARR_HNUM;i++) begin
        assign gbus_load_kgen_addr[i] = {1'b0, gbus_load_kgen_addr_tmp[i]};
    end
endgenerate


//KGEN inst, load weight state, gbus_wen logic
generate
    for(i=0;i<`ARR_HNUM;i++) begin
        for(j=0;j<`ARR_VNUM;j++) begin
            always_comb begin
                if(kload_core_row_cnt==i && kload_core_col_cnt==j)
                    gbus_load_kgen_wen[i][j] = 1'b1;
                else
                    gbus_load_kgen_wen[i][j] = 1'b0;
            end
        end
    end
endgenerate

//KGEN inst, load weight state, global sram read addr
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        global_sram_load_kgen_addr <= `GLOBAL_SRAM_LOAD_KGEN_BASE_ADDR;
    end
    else if(kload_core_inc) begin //global sram raddr for vgen load weight
        global_sram_load_kgen_addr <= global_sram_load_kgen_addr + 1;
    end
end 

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        kload_core_inc_d<='0;
    end
    else begin
        kload_core_inc_d <= kload_core_inc;
    end
end


/******kGEN Compute address generation******/
logic act_kgen_reuse_finish;
logic lbuf_kcompute_raddr_inc;
logic kcompute_core_en;
logic abuf_kcompute_raddr_inc;

assign kcompute_core_en = (inst_reg[pc_reg]==K_GEN) && (state==COMPUTE);
assign abuf_kcompute_raddr_inc = kcompute_core_en && (!abuf_almost_full[0][0]) && (global_sram_compute_kgen_addr < ABUF_CNT);
assign lbuf_kcompute_raddr_inc = kcompute_core_en && (!lbuf_almost_full[0][0]) && ((lbuf_kcompute_load_num < (LBUF_KGEN_LOAD_NUM-1)) || (lbuf_kcompute_load_num == (LBUF_KGEN_LOAD_NUM-1)) && (kcompute_wei_cnt < (WEI_KGEN_NUM-1))) ;

assign kcompute_finish = kcompute_wb_col_overflow;

assign act_kgen_reuse_finish = (kcompute_act_reuse_cnt == ACT_REUSE_KGEN_NUM-1);

// Each column shares the same cmem read address & cmem read enable
assign cmem_compute_kgen_raddr_tmp = kcompute_wei_cnt;
generate
    for(i=0;i<`ARR_HNUM;i++) begin
        assign cmem_compute_kgen_raddr[i][0] = {1'b0, cmem_compute_kgen_raddr_tmp};
        assign gbus_compute_kgen_addr[i] = {1'b1, gbus_compute_kgen_addr_tmp};
    end
endgenerate

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            for(int j=1;j<`ARR_VNUM;j++) begin
                cmem_compute_kgen_raddr[i][j] <= 0;
            end
        end
    end
    else begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            for(int j=1;j<`ARR_VNUM;j++) begin
                cmem_compute_kgen_raddr[i][j] <= cmem_compute_kgen_raddr[i][j-1];
            end
        end
    end
end

// cmem read enable

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        cmem_compute_kgen_ren <= '0;
    end
    else begin
        for(int i=0;i<`ARR_HNUM;i++) begin
            for(int j=0;j<`ARR_VNUM;j++) begin
                if(j==0) begin
                    cmem_compute_kgen_ren[i][j] <= lbuf_kcompute_raddr_inc;
                end
                else begin
                    cmem_compute_kgen_ren[i][j] <= cmem_compute_kgen_ren[i][j-1];
                end
            end
        end
    end
end

// Load abuf from SRAM for each row simultaneously, if !act_kgen_reuse_finish, then stop loading, using abuf_reuse_en
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        global_sram_compute_kgen_addr <= `GLOBAL_SRAM_COMPUTE_QGEN_BASE_ADDR;
    end
    else if(kcompute_core_en) begin
        if(global_sram_compute_kgen_ren)
            global_sram_compute_kgen_addr <= global_sram_compute_kgen_addr + 1;
        else
            global_sram_compute_kgen_addr <= global_sram_compute_kgen_addr;
    end
    else
        global_sram_compute_kgen_addr <= `GLOBAL_SRAM_COMPUTE_QGEN_BASE_ADDR;
end 

assign global_sram_compute_kgen_ren = abuf_kcompute_raddr_inc;

// Hlink wen in the next cycle of MAC & quantization of the first column
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn)
        hlink_compute_kgen_wen <= '0;
    else begin
        for(int i=0; i<`ARR_HNUM; i++)
            hlink_compute_kgen_wen[i] <= abuf_kcompute_raddr_inc;
    end
end

// MAC & quantization, if kcompute_act_reuse_cnt == `ACT_REUSE_NUM - 1 (act_kgen_reuse_finish == 1), then ren, otherwise ruse_en
always_comb begin
    for(int i=0; i<`ARR_HNUM; i++) begin
        abuf_compute_kgen_ren[i][0] = act_kgen_reuse_finish && (!abuf_empty[i][0]) && (!lbuf_empty[i][0]) ? 1 : 0;
        abuf_compute_kgen_reuse_ren[i][0] = (!act_kgen_reuse_finish) && (!abuf_reuse_empty[i][0]) && (!lbuf_empty[i][0]) ? 1 : 0;
        abuf_compute_kgen_reuse_rst[i][0] = ((kcompute_act_cnt == (ACT_KGEN_NUM-1)) && abuf_reuse_ren[i][0]) ? 1 : 0;
        lbuf_compute_kgen_ren[i][0] = abuf_compute_kgen_ren[i][0] || abuf_compute_kgen_reuse_ren[i][0];
    end
end

assign lbuf_compute_kgen_reuse_ren = '0;
assign lbuf_compute_kgen_reuse_rst = '0;

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                abuf_compute_kgen_ren[j][i] <= 0;
                lbuf_compute_kgen_ren[j][i] <= 0;
                abuf_compute_kgen_reuse_ren[j][i] <= 0;
                abuf_compute_kgen_reuse_rst[j][i] <= 0;
            end
        end
    end
    else begin
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                abuf_compute_kgen_ren[j][i] <= abuf_compute_kgen_ren[j][i-1];
                lbuf_compute_kgen_ren[j][i] <= lbuf_compute_kgen_ren[j][i-1];
                abuf_compute_kgen_reuse_ren[j][i] <= abuf_compute_kgen_reuse_ren[j][i-1];
                abuf_compute_kgen_reuse_rst[j][i] <= abuf_compute_kgen_reuse_rst[j][i-1];
            end
        end
    end
end 

// Write back to K cache

// Decide which col to write back
always_comb begin
    for(int i=0; i<`ARR_HNUM; i++) begin
        for(int j=0; j<`ARR_VNUM; j++) begin
            if((j == kcompute_wb_col_cnt) && (|gbus_rvalid)) begin
                gbus_compute_kgen_wen[i][j] = 1;
            end
            else gbus_compute_kgen_wen[i][j] = 0;
        end
    end
end 

// Writing back address
assign gbus_compute_kgen_addr_tmp = kcompute_wb_cnt + (`CMEM_KBASE_ADDR);


// Read channel is not useful here
assign gbus_compute_kgen_ren = '0;



    //////////////////////////////////////////////////
    //                                              //
    //                QGEN                          //
    //                                              //
    //////////////////////////////////////////////////


assign qload_core_inc=(inst_reg[pc_reg] == Q_GEN) && (state == LOAD_WEIGHT) && !qload_core_row_overflow;
//load_finish signal
// assign qload_finish=qload_core_row_overflow;
assign qload_finish = 1;

//QGEN inst, load weight state, gbus_addr logic
always_comb begin
    for(int i=0;i<`ARR_HNUM;i++) begin
        if(i==qload_core_row_cnt)
            gbus_load_qgen_addr_tmp[i] = qload_in_core_cnt;
        else
            gbus_load_qgen_addr_tmp[i]='0;
    end
end

generate
    for(i=0;i<`ARR_HNUM;i++) begin
        assign gbus_load_qgen_addr[i] = {1'b0, gbus_load_qgen_addr_tmp[i]};
    end
endgenerate

//QGEN inst, load weight state, gbus_wen logic

generate
    for(i=0;i<`ARR_HNUM;i++) begin
        for(j=0;j<`ARR_VNUM;j++) begin
            always_comb begin
                if(qload_core_row_cnt==i && qload_core_col_cnt==j)
                    gbus_load_qgen_wen[i][j] = 1'b1;
                else
                    gbus_load_qgen_wen[i][j] = 1'b0;
            end
        end
    end
endgenerate

//QGEN inst, load weight state, global sram read addr
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        global_sram_load_qgen_addr <= `GLOBAL_SRAM_LOAD_QGEN_BASE_ADDR;
    end
    else if(qload_core_inc) begin //global sram raddr for vgen load weight
        global_sram_load_qgen_addr <= global_sram_load_qgen_addr + 1;
    end
end 

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        qload_core_inc_d<='0;
    end
    else begin
        qload_core_inc_d <= qload_core_inc;
    end
end

/******QGEN Compute address generation******/
logic act_qgen_reuse_finish;
logic lbuf_qcompute_raddr_inc;
logic qcompute_core_en;
logic abuf_qcompute_raddr_inc;
assign qcompute_core_en = (inst_reg[pc_reg] == Q_GEN) && (state==COMPUTE);
assign abuf_qcompute_raddr_inc = qcompute_core_en && (!abuf_almost_full[0][0]) && (global_sram_compute_qgen_addr < ABUF_CNT);
assign lbuf_qcompute_raddr_inc = qcompute_core_en && (!lbuf_almost_full[0][0]) && ((lbuf_qcompute_load_num < (LBUF_QGEN_LOAD_NUM-1)) || (lbuf_qcompute_load_num == (LBUF_QGEN_LOAD_NUM-1)) && (qcompute_wei_cnt < (WEI_QGEN_NUM-1)));

assign act_qgen_reuse_finish = (qcompute_act_reuse_cnt == ACT_REUSE_QGEN_NUM-1);
assign qcompute_finish = (global_sram_compute_qgen_waddr == (`GLOBAL_SRAM_COMPUTE_QGEN_BASE_ADDR + WB_QGEN_NUM)) && wbfifo_empty2controller;

assign cmem_compute_qgen_raddr_tmp = qcompute_wei_cnt;

generate
    for(i = 0; i < `ARR_HNUM; i++) begin
        assign cmem_compute_qgen_raddr[i][0] = {1'b0, cmem_compute_qgen_raddr_tmp};
    end
endgenerate

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=0; i<`ARR_HNUM;i++) begin
            for(int j=1;j<`ARR_VNUM;j++) begin
                cmem_compute_qgen_raddr[i][j] <= 0;
            end
        end
    end
    else begin
        for(int i=0; i<`ARR_HNUM;i++) begin
            for(int j=1;j<`ARR_VNUM;j++) begin
                cmem_compute_qgen_raddr[i][j] <= cmem_compute_qgen_raddr[i][j-1];
            end
        end
    end
end


always_ff@(posedge clk, negedge rstn) begin

    if(!rstn) begin
        cmem_compute_qgen_ren <= '0;
    end
    else begin
        for(int i=0; i< `ARR_HNUM;i++) begin
            for(int j=0; j<`ARR_VNUM;j++) begin
                if(j==0) begin
                    cmem_compute_qgen_ren[i][j] <= lbuf_qcompute_raddr_inc;
                end
                else begin
                    cmem_compute_qgen_ren[i][j] <= cmem_compute_qgen_ren[i][j-1];
                end
            end
        end
    end

end

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        global_sram_compute_qgen_addr <= `GLOBAL_SRAM_COMPUTE_QGEN_BASE_ADDR;
    end
    else if(qcompute_core_en) begin
        if(global_sram_compute_qgen_ren) begin
            global_sram_compute_qgen_addr <= global_sram_compute_qgen_addr + 1;
        end
        else begin
            global_sram_compute_qgen_addr <= global_sram_compute_qgen_addr;
        end
    end
    else begin
        global_sram_compute_qgen_addr <= `GLOBAL_SRAM_COMPUTE_QGEN_BASE_ADDR;
    end
end

assign global_sram_compute_qgen_ren = abuf_qcompute_raddr_inc;

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        hlink_compute_qgen_wen <= '0;
    end
    else begin
        
        for(int i = 0; i < `ARR_HNUM; i++) begin

            hlink_compute_qgen_wen[i] <= abuf_qcompute_raddr_inc;
        end
    end
end

//MAC and quantization, if qcompute_act_reuse_cnt == `ACT_REUSE_NUM-1 (act_qgen_reuse_finish==1), then ren, otherwise reuse_en
generate
    for(i=0; i<`ARR_HNUM; i++) begin
        assign abuf_compute_qgen_ren[i][0] = act_qgen_reuse_finish && (!abuf_empty[i][0]) && (!lbuf_empty[i][0]) ? 1 : 0;
        assign abuf_compute_qgen_reuse_ren[i][0] = (!act_qgen_reuse_finish) && (!abuf_reuse_empty[i][0]) && (!lbuf_empty[i][0]) ? 1 : 0;
        assign abuf_compute_qgen_reuse_rst[i][0] = ((qcompute_act_cnt == (ACT_QGEN_NUM-1)) && abuf_reuse_ren[i][0]) ? 1 : 0;
        assign lbuf_compute_qgen_ren[i][0] = (abuf_compute_qgen_ren[i][0] || abuf_compute_qgen_reuse_ren[i][0]);
    end
endgenerate

assign lbuf_compute_qgen_reuse_ren = '0;
assign lbuf_compute_qgen_reuse_rst = '0;

always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                abuf_compute_qgen_ren[j][i] <= 0;
                lbuf_compute_qgen_ren[j][i] <= 0;
                abuf_compute_qgen_reuse_ren[j][i] <= 0;
                abuf_compute_qgen_reuse_rst[j][i] <= 0;
            end
        end
    end
    else begin
        for(int i=1; i<`ARR_VNUM; i++) begin
            for(int j=0; j<`ARR_HNUM; j++) begin
                abuf_compute_qgen_ren[j][i] <= abuf_compute_qgen_ren[j][i-1];
                lbuf_compute_qgen_ren[j][i] <= lbuf_compute_qgen_ren[j][i-1];
                abuf_compute_qgen_reuse_ren[j][i] <= abuf_compute_qgen_reuse_ren[j][i-1];
                abuf_compute_qgen_reuse_rst[j][i] <= abuf_compute_qgen_reuse_rst[j][i-1];
            end
        end
    end
end

// Write results of quantization to global SRAM

 always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        global_sram_compute_qgen_waddr <= `GLOBAL_SRAM_COMPUTE_QGEN_BASE_WADDR;
    end
    else if(gbus_rvalid && qcompute_core_en) begin //global sram raddr for qcompute writing back results
        global_sram_compute_qgen_waddr <= global_sram_compute_qgen_waddr + 1;
//         qcompute_finish <= 1'b1;
    end
    else begin
        global_sram_compute_qgen_waddr <= global_sram_compute_qgen_waddr;
//         qcompute_finish <= 1'b0;
    end
end

assign global_sram_compute_qgen_wen = |gbus_rvalid && qcompute_core_en;



    //////////////////////////////////////////////////
    //                                              //
    //                Attention                     //
    //                                              //
    //////////////////////////////////////////////////


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
                    if(~abuf_almost_full[i][0]) begin//head's abuf is not full
                        att_head_inc=1'b1;
                        next_hlink_att_wen[i] = 1'b1;
                        next_att_hlink_sel = GSRAM02HLINK; //FIXME: need pingpong flag to decide which GSRAM
                    end
                end
            end
            else begin// saddr: write back from consmax, using last head's consmax output valid signal as indicator
                if(~abuf_almost_full[i][0] & ctrl_cons_ovalid[i]) begin //FIXME: When abuf full, we will lose the data from consmax
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
                if(~abuf_almost_full[i][0] & ~att_interleave_flag) begin//head's abuf is not full
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
parameter CMEM_ATT_KADDR_CNT = (`SEQ_LENGTH/`ARR_VNUM*`N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT));//K addr space needed for 1 row of Q vector in one core
parameter CMEM_ATT_VADDR_CNT = (`N_MODEL/`N_HEAD/`ARR_VNUM*`SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT));//V addr space needed for 1 row of V vector in one core
logic [$clog2(CMEM_ATT_KADDR_CNT)-1:0] att_cmem_kaddr_cnt;
logic att_cmem_kaddr_inc;
logic att_cmem_kaddr_overflow;
logic [$clog2(CMEM_ATT_VADDR_CNT)-1:0] att_cmem_vaddr_cnt;
logic att_cmem_vaddr_inc;
logic att_cmem_vaddr_overflow;
counter #(CMEM_ATT_KADDR_CNT-1) att_cmemk_counter (.clk(clk),.rstn(rstn),.inc(att_cmem_kaddr_inc),.overflow(att_cmem_kaddr_overflow),.out(att_cmem_kaddr_cnt));
counter #(CMEM_ATT_VADDR_CNT-1) att_cmemv_counter (.clk(clk),.rstn(rstn),.inc(att_cmem_vaddr_inc),.overflow(att_cmem_vaddr_overflow),.out(att_cmem_vaddr_cnt));

logic att_cmem_rd_interleave;
logic att_cmem_k_start;
logic att_cmem_v_start;

assign att_cmem_kaddr_inc = att_en & ~att_cmem_rd_interleave & !lbuf_almost_full[`ARR_HNUM-1][`ARR_VNUM-1] &~att_compute_finish;
assign att_cmem_vaddr_inc = att_en & att_cmem_rd_interleave & !lbuf_almost_full[`ARR_HNUM-1][`ARR_VNUM-1] &~att_compute_finish;

//when abuf start loading q or p, start loading k or v
always_ff @(posedge clk or negedge rstn) begin
    if(~rstn)begin
        att_cmem_rd_interleave <=0;
    end

    // cmem read interleave flag between k and v
    else if(att_cmem_kaddr_overflow | att_cmem_vaddr_overflow) begin
        att_cmem_rd_interleave <= ~att_cmem_rd_interleave;
    end
end

always_comb begin
    for(int i=0;i<`ARR_HNUM;i++) begin
        for(int j=0;j<`ARR_VNUM;j++) begin
            att_cmem.cmem_wen = 1'b0;
            att_cmem.cmem_waddr = 'b0;
            att_cmem.cmem_ren = 1'b0;
            att_cmem.cmem_raddr = {1'b1,{(`ARR_GBUS_ADDR-1){1'b0}}};
        end
    end

    for(int i=0;i<`ARR_HNUM;i++) begin
        for(int j=0;j<`ARR_VNUM;j++) begin
            if(att_cmem_kaddr_inc) begin
                att_cmem.cmem_ren[i][j] = 1'b1;
                att_cmem.cmem_raddr[i][j][`ARR_GBUS_ADDR-2:0] = att_cmem_kaddr_cnt + `CMEM_KBASE_ADDR;
            end
            else if(att_cmem_vaddr_inc) begin
                att_cmem.cmem_ren[i][j] = 1'b1;
                att_cmem.cmem_raddr[i][j][`ARR_GBUS_ADDR-2:0] = att_cmem_vaddr_cnt + `CMEM_VBASE_ADDR;
            end
        end
    end
end

/******Compute: ABUF Reuse and Ren, LBUF Ren******/
//cfg_acc_num gen
always_ff @(posedge clk or negedge rstn) begin
    if(!rstn) begin
        att_cfg_acc_num<='0;
    end
    else begin
        for(int i=0;i<`ARR_HNUM; i++) begin
            if(att_p_abuf_ren[i][0] | att_p_abuf_reuse_ren[i][0]) begin
                att_cfg_acc_num[i]<= (`N_MODEL/`N_HEAD/(`ARR_GBUS_DATA/`ARR_IDATA_BIT) - 1);
            end
            else if(att_q_abuf_ren[i][0] | att_q_abuf_reuse_ren[i][0]) begin
                att_cfg_acc_num[i]<= (`SEQ_LENGTH/(`ARR_GBUS_DATA/`ARR_IDATA_BIT)-1);
            end
        end
    end
end
/******Attention QK Logic******/
assign att_qk_inc = (!abuf_reuse_empty[0][0]) && (!lbuf_empty[0][0]) && ~att_compute_interleave_flag && att_en && ~att_compute_finish; //att_en start signal, att_compute_finish is finish signal 
assign att_qk_reuse_loop_inc = att_qk_cnt_overflow;
assign att_q_abuf_reuse_finish = (att_qk_reuse_loop_cnt == ATT_QK_LOOP-1);

generate
    for(genvar i=0; i<`ARR_HNUM; i++) begin
        for(genvar j=0;j<`ARR_VNUM;j++) begin
            if(i==0 & j==0) begin
                assign att_q_abuf_ren[0][0] = att_q_abuf_reuse_finish && (!abuf_empty[0][0]) && (!lbuf_empty[0][0]);
                assign att_q_abuf_reuse_ren[0][0] = (!att_q_abuf_reuse_finish) && (!abuf_reuse_empty[0][0]) && (!lbuf_empty[0][0]) && ~att_compute_interleave_flag && att_en && ~att_compute_finish; 
                assign att_q_abuf_reuse_rst[0][0] = ((att_qk_cnt == ATT_QK_CNT-1) && att_q_abuf_reuse_ren[0][0]);
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
        for(genvar j=0;j<`ARR_VNUM;j++) begin
            if(i==0 & j==0) begin
                assign att_p_abuf_ren[0][0] = att_p_abuf_reuse_finish && (!abuf_empty[0][0]) && (!lbuf_empty[0][0]);
                assign att_p_abuf_reuse_ren[0][0] = (!att_p_abuf_reuse_finish) && (!abuf_reuse_empty[0][0]) && (!lbuf_empty[0][0]) && att_compute_interleave_flag && att_en && ~att_compute_finish; 
                assign att_p_abuf_reuse_rst[0][0] = ((att_pv_cnt == ATT_PV_CNT-1) && att_p_abuf_reuse_ren[0][0]);
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

    //////////////////////////////////////////////////
    //                                              //
    //                FFN0 FFN1                     //
    //                                              //
    //////////////////////////////////////////////////

    //////////////////////////////////////////////////
    //                                              //
    //                Counters                      //
    //                                              //
    //////////////////////////////////////////////////


//============================== Counters for loading weight =================================

logic load_core_inc;
logic load_core_col_inc;
logic load_core_row_inc;
logic load_in_core_overflow;
logic load_core_col_overflow;
logic load_core_row_overflow;

logic [LOAD_INCORE_WIDTH-1:0] load_in_core_cnt;
logic [BIT_WIDTH_V-1:0] load_core_col_cnt;
logic [BIT_WIDTH_H-1:0] load_core_row_cnt;

counter #(CORE_ADDR_CNT-1) load_in_core_counter (.clk(clk),.rstn(rstn),.inc(load_core_inc),.overflow(load_in_core_overflow),.out(load_in_core_cnt));
counter #(`ARR_VNUM-1) load_core_col_counter (.clk(clk),.rstn(rstn),.inc(load_core_col_inc),.overflow(load_core_col_overflow),.out(load_core_col_cnt));
counter #(`ARR_HNUM-1) load_core_row_counter (.clk(clk),.rstn(rstn),.inc(load_core_row_inc),.overflow(load_core_row_overflow),.out(load_core_row_cnt));

assign load_core_inc = vload_core_inc_d || kload_core_inc_d || qload_core_inc_d;
assign load_core_col_inc = (vload_in_core_overflow & vload_core_inc_d) || (kload_in_core_overflow & kload_core_inc_d) || (qload_in_core_overflow & qload_core_inc_d);
assign load_core_row_inc = load_core_col_inc & (vload_core_col_overflow || kload_core_col_overflow || qload_core_col_overflow);

assign vload_in_core_overflow = vload_core_inc_d ? load_in_core_overflow : '0;
assign vload_core_col_overflow = vload_core_inc_d ? load_core_col_overflow : '0;
assign vload_core_row_overflow = vload_core_inc_d ? load_core_row_overflow : '0;
assign vload_in_core_cnt = vload_core_inc_d ? load_in_core_cnt : '0;
assign vload_core_col_cnt = vload_core_inc_d ? load_core_col_cnt : '0;
assign vload_core_row_cnt = vload_core_inc_d ? load_core_row_cnt : '0;

assign kload_in_core_overflow = kload_core_inc_d ? load_in_core_overflow : '0;
assign kload_core_col_overflow = kload_core_inc_d ? load_core_col_overflow : '0;
assign kload_core_row_overflow = kload_core_inc_d ? load_core_row_overflow : '0;
assign kload_in_core_cnt = kload_core_inc_d ? load_in_core_cnt : '0;
assign kload_core_col_cnt = kload_core_inc_d ? load_core_col_cnt : '0;
assign kload_core_row_cnt = kload_core_inc_d ? load_core_row_cnt : '0;

assign qload_in_core_overflow = qload_core_inc_d ? load_in_core_overflow : '0;
assign qload_core_col_overflow = qload_core_inc_d ? load_core_col_overflow : '0;
assign qload_core_row_overflow = qload_core_inc_d ? load_core_row_overflow : '0;
assign qload_in_core_cnt = qload_core_inc_d ? load_in_core_cnt : '0;
assign qload_core_col_cnt = qload_core_inc_d ? load_core_col_cnt : '0;
assign qload_core_row_cnt = qload_core_inc_d ? load_core_row_cnt : '0;
    
//============================== Counters for Vgen =================================

    always_ff @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            vcompute_act_inc <= 0;
        end
        else begin
            vcompute_act_inc <= abuf_vcompute_raddr_inc;
        end
    end

    assign vcompute_wb_col_inc = vcompute_core_en && (|gbus_rvalid);
    // counter for loading abuf
    // ACT_NUM: NUmber of activations in one single loop
    counter #(ACT_VGEN_NUM-1) vcompute_act_counter (.clk(clk),.rstn(rstn),.inc(abuf_vcompute_raddr_inc),.overflow(vcompute_act_overflow),.out(vcompute_act_cnt));
    logic vcompute_act_num_overflow;
    counter #(ABUF_VGEN_LOAD_NUM) vcompute_act_num_counter(.clk(clk), .rstn(rstn), .inc(vcompute_act_overflow), .overflow(vcompute_act_num_overflow), .out(abuf_vcompute_load_num));


    // counter for loading lbuf
    counter #(WEI_VGEN_NUM-1) vcompute_wei_counter (.clk(clk),.rstn(rstn),.inc(vcompute_core_en && (!abuf_empty[0][0]) && (!lbuf_reuse_empty[0][0])),.overflow(vcompute_wei_overflow),.out(vcompute_wei_cnt));
    // REUSE_NUM: Number of reuse times 
    counter #(WEI_REUSE_VGEN_NUM-1) vcompute_wei_reuse_counter (.clk(clk),.rstn(rstn),.inc(vcompute_wei_overflow),.overflow(vcompute_wei_reuse_overflow),.out(vcompute_wei_reuse_cnt));

    // counter for writing back 
    assign vcompute_wb_inc = vcompute_wb_col_inc & vcompute_wb_col_overflow;
    
    // Decide which core to write back
    counter #(`ARR_VNUM-1) vcompute_wb_col_counter (.clk(clk),.rstn(rstn),.inc(vcompute_wb_col_inc),.overflow(vcompute_wb_col_overflow),.out(vcompute_wb_col_cnt));

    // Generate address for writing back
    counter #(WB_VGEN_NUM-1) vcompute_wb_counter (.clk(clk),.rstn(rstn),.inc(vcompute_wb_inc),.overflow(vcompute_wb_overflow),.out(vcompute_wb_cnt));


//============================== Counters for Kgen =================================

    always_ff @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            kcompute_wei_inc <= 0;
        end
        else begin
            kcompute_wei_inc <= lbuf_kcompute_raddr_inc;
        end
    end

    assign kcompute_wb_inc = kcompute_core_en && (|gbus_rvalid);
    // counter for loading abuf
    // ACT_KGEN_NUM: NUmber of activations in one single loop
    counter #(ACT_KGEN_NUM-1) kcompute_act_counter (.clk(clk),.rstn(rstn),.inc(kcompute_core_en && (!abuf_reuse_empty[0][0]) && (!lbuf_empty[0][0])),.overflow(kcompute_act_overflow),.out(kcompute_act_cnt));
    // REUSE_NUM: Number of reuse times 
    counter #(ACT_REUSE_KGEN_NUM-1) kcompute_act_reuse_counter (.clk(clk),.rstn(rstn),.inc(kcompute_act_overflow),.overflow(kcompute_act_reuse_overflow),.out(kcompute_act_reuse_cnt));

    // counter for loading lbuf
    // WEI_KGEN_NUM
    counter #(WEI_KGEN_NUM-1) kcompute_wei_counter (.clk(clk),.rstn(rstn),.inc(kcompute_wei_inc),.overflow(kcompute_wei_overflow),.out(kcompute_wei_cnt));
    logic kcompute_wei_num_overflow;
    counter #(LBUF_KGEN_LOAD_NUM) kcompute_wei_num_counter(.clk(clk), .rstn(rstn), .inc(kcompute_wei_overflow), .overflow(kcompute_wei_num_overflow), .out(lbuf_kcompute_load_num));

    // counter for writing back 
    assign kcompute_wb_col_inc = kcompute_wb_inc & kcompute_wb_overflow;

    // Generate address for writing back
    counter #(WB_KCOL_CNT-1) kcompute_wb_counter (.clk(clk),.rstn(rstn),.inc(kcompute_wb_inc),.overflow(kcompute_wb_overflow),.out(kcompute_wb_cnt));

    // Decide which core to write back
    counter #(`ARR_VNUM-1) kcompute_wb_col_counter (.clk(clk),.rstn(rstn),.inc(kcompute_wb_col_inc),.overflow(kcompute_wb_col_overflow),.out(kcompute_wb_col_cnt));

//============================== Counters for Qgen =================================

     always_ff @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            qcompute_wei_inc <= 0;
        end
        else begin
            qcompute_wei_inc <= lbuf_qcompute_raddr_inc;
        end
    end

    counter #(ACT_QGEN_NUM-1) qcompute_act_counter (.clk(clk),.rstn(rstn),.inc(qcompute_core_en && (!abuf_reuse_empty[0][0]) && (!lbuf_empty[0][0])),.overflow(qcompute_act_overflow),.out(qcompute_act_cnt));
    // REUSE_NUM: Number of reuse times 
    counter #(ACT_REUSE_QGEN_NUM-1) qcompute_act_reuse_counter (.clk(clk),.rstn(rstn),.inc(qcompute_act_overflow),.overflow(qcompute_act_reuse_overflow),.out(qcompute_act_reuse_cnt));

    counter #(WEI_QGEN_NUM-1) qcompute_wei_counter (.clk(clk),.rstn(rstn),.inc(qcompute_wei_inc),.overflow(qcompute_wei_overflow),.out(qcompute_wei_cnt));
    logic qcompute_wei_num_overflow;
    counter #(LBUF_QGEN_LOAD_NUM) qcompute_wei_num_counter(.clk(clk), .rstn(rstn), .inc(qcompute_wei_overflow), .overflow(qcompute_wei_num_overflow), .out(lbuf_qcompute_load_num));

//============================== Counters for Attention =================================

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
    
    //writeback
    counter #(ATT_OUT_WB-1) att_wb_out_counter (.clk(clk),.rstn(rstn),.inc(att_wb_out_inc),.overflow(att_wb_out_overflow),.out(att_wb_out_cnt));
    counter #(VEC_ATT_SADDR_CNT-1) att_cons_out_counter (.clk(clk),.rstn(rstn),.inc(att_cons_out_inc),.overflow(att_cons_out_overflow),.out(att_cons_out_cnt));
    counter #(`SEQ_LENGTH-1) att_wb_seq_counter (.clk(clk),.rstn(rstn),.inc(att_wb_seq_inc),.overflow(att_wb_seq_overflow),.out(att_wb_seq_cnt));

endmodule
