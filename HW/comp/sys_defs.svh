`ifndef __SYS_DEFS_SVH__
`define __SYS_DEFS_SVH__
`define INTC_NO_PWR_PINS
`timescale 1ps/1ps  
`define CLK_CFG 3
    //////////////////////////////////////////////////
    //                                              //
    //              Layer parameters                //
    //                                              //
    //////////////////////////////////////////////////
`define N_HEAD  6
`define N_MODEL 384 
`define SEQ_LENGTH 64 //key matrix rows

//Q,K,V: (Block_size,N_MODEL/N_HEAD)
//Q,K: store row-wise inside the core, 1 row Q/k, N_MODEL/N_HEAD
    //////////////////////////////////////////////////
    //                                              //
    //                Core Array                    //
    //                                              //
    //////////////////////////////////////////////////
`define ARR_HNUM    6
`define ARR_VNUM    8
`define ARR_GBUS_ADDR   12
`define ARR_GBUS_DATA   64

`define ARR_CDATA_BIT   8
`define ARR_ODATA_BIT   16
`define ARR_IDATA_BIT   8
`define ARR_MAC_NUM   8


`define ARR_LBUF_DEPTH  64
`define ARR_LBUF_DATA   8*`ARR_MAC_NUM
`define ARR_LBUF_ADDR   $clog2(`ARR_LBUF_DEPTH)
`define ARR_WMEM_DEPTH   512
`define ARR_CACHE_DEPTH  256
`define GLOBAL_SRAM_DEPTH  4096

`define ARR_CONSLUT_ADDR (`ARR_IDATA_BIT>>1)
`define ARR_CONSLUT_DATA (7+8+1)

`define LN_FP_W            16    

`define INST_REG_DEPTH     4

//CTRL
typedef logic [`ARR_HNUM-1:0][`ARR_VNUM-1:0]                CTRL;

//GBUS, VLINK, HLINK, CMEM
typedef logic [`ARR_HNUM-1:0][`ARR_VNUM-1:0][`ARR_GBUS_DATA-1:0] G_DATA;
typedef logic [`ARR_HNUM-1:0][`ARR_VNUM-1:0][`ARR_GBUS_ADDR-1:0] G_ADDR;

//LBUF, ABUF
typedef logic [`ARR_HNUM-1:0][`ARR_VNUM-1:0][`ARR_LBUF_DATA-1:0] BUF_DATA;
typedef logic [`ARR_HNUM-1:0][`ARR_VNUM-1:0][`ARR_LBUF_ADDR-1:0] BUF_ADDR;

// Channel - Global Bus to Access Core Memory and MAC Result
// 1. Write Channel
//      1.1 Chip Interface -> WMEM for Weight Upload
//      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
//      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
// 2. Read Channel
//      2.1 WMEM       -> Chip Interface for Weight Check
//      2.2 KV Cache   -> Chip Interface for KV Checnk
//      2.3 MAC Result -> Vector Engine  for Post Processing
typedef struct packed {
    G_ADDR gbus_addr;
    CTRL   gbus_wen;
    G_DATA gbus_wdata;
    CTRL   gbus_ren;
    G_DATA gbus_rdata;     // To Chip Interface (Debugging) and Vector Engine (MAC)
    CTRL   gbus_rvalid;
} GBUS_ARR_PACKET;

typedef struct packed {
    CTRL    vlink_enable;
    G_DATA  vlink_wdata;
    CTRL    vlink_wen;
    G_DATA  vlink_rdata;
    CTRL    vlink_rvalid;
} VLINK_ARR_PACKET;

typedef struct packed {
    G_DATA  hlink_wdata;    //hlink_wdata go through reg, to hlink_rdata
    CTRL    hlink_wen;
    G_DATA  hlink_rdata;
    CTRL    hlink_rvalid;
} HLINK_ARR_PACKET;

typedef struct packed {
    G_ADDR  cmem_waddr;      // Write Value to KV Cache, G_BUS -> KV Cache, debug.
    CTRL    cmem_wen;
    G_ADDR  cmem_raddr;
    CTRL    cmem_ren;
} CMEM_ARR_PACKET;

typedef struct packed {
    CTRL        lbuf_empty;
    CTRL        lbuf_full;
    CTRL        lbuf_ren;
} LBUF_ARR_PACKET;

typedef struct packed {
    CTRL        abuf_empty;
    CTRL        abuf_full;
    CTRL        abuf_ren;
} ABUF_ARR_PACKET;

typedef struct packed {
    // logic       [`ARR_CDATA_BIT-1:0] cfg_acc_num;//DELETE THIS
    logic       [`ARR_ODATA_BIT-1:0] cfg_quant_scale;//16
    logic       [`ARR_ODATA_BIT-1:0] cfg_quant_bias;//10
    logic       [`ARR_ODATA_BIT-1:0] cfg_quant_shift;//2
    logic       [`ARR_CDATA_BIT-1:0] cfg_consmax_shift;//0

} CFG_ARR_PACKET;


    //////////////////////////////////////////////////
    //                                              //
    //                Controller                    //
    //                                              //
    //////////////////////////////////////////////////
`define GLOBAL_SRAM_LOAD_VGEN_BASE_ADDR '0
`define GLOBAL_SRAM_COMPUTE_VGEN_BASE_ADDR '0
`define GLOBAL_SRAM_LOAD_QGEN_BASE_ADDR '0
`define GLOBAL_SRAM_COMPUTE_QGEN_BASE_ADDR '0
`define GLOBAL_SRAM_COMPUTE_QGEN_BASE_WADDR '0
`define GLOBAL_SRAM_LOAD_KGEN_BASE_ADDR '0
`define GLOBAL_SRAM_COMPUTE_KGEN_BASE_ADDR '0

`define CMEM_KBASE_ADDR 0
`define CMEM_VBASE_ADDR 128

//Attention base addr
//base read addr for Q during attention
`define GSRAM_ATT_QBASE_ADDR 0

typedef enum logic [3:0] {
    NOP     = 4'h0,
    Q_GEN   = 4'h1,
    K_GEN   = 4'h2,
    V_GEN   = 4'h3,
    ATT     = 4'h4
} GPT_COMMAND;

typedef enum logic [2:0] {
    IDLE        = 3'h0,
    LOAD_WEIGHT = 3'h1,
    COMPUTE   = 3'h2,
    FINISH = 3'h3,
    COMPLETE = 3'h4
} STATE;

    //////////////////////////////////////////////////
    //                                              //
    //                Global SRAM                   //
    //                                              //
    //////////////////////////////////////////////////

typedef struct packed {
    logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]global_sram_waddr;
    logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0]global_sram_raddr;
    logic global_sram_wen;
    logic [`ARR_GBUS_DATA-1:0] global_sram_wdata;
    logic global_sram_ren;
    logic [`ARR_GBUS_DATA-1:0] global_sram_rdata;
} GLOBAL_SRAM_PACKET;

typedef enum logic [1:0] {
    LN2HLINK = 2'h0,
    CONS2HLINK = 2'h1,
    GSRAM02HLINK = 2'h2,
    GSRAM12HLINK = 2'h3
} HLINK_WSEL;

typedef enum logic [1:0] {
    GBUS2GBUS = 2'h0,
    SPI2GBUS = 2'h1
} GBUS_WSEL;

typedef enum logic [1:0] {
    // GBUS2GSRAM = 2'h0, //gbus
    FIFO2GSRAM = 2'h0, //gbus
    LN2GSRAM  = 2'h1,   //layernorm
    SPI2GSRAM = 2'h2,
    WSEL_DISABLE = 2'h3
} GSRAM_WSEL;

typedef enum logic [1:0] {
    // GBUS2GSRAM = 2'h0, //gbus
    GSRAM2CHIP = 2'h0, //gbus
    GSRAM2SPI  = 2'h1,   //layernorm
    RSEL_DISABLE = 2'h2
} GSRAM_RSEL;

typedef enum logic [1:0] {
    GSRAM02LN = 2'h0,
    GSRAM12LN = 2'h1
} LN_WSEL;

`endif
