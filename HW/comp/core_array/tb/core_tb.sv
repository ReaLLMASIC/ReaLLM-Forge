module core_tb
    // 1. Global Bus and Core-to-Core Link
    parameter   GBUS_DATA   = 64;               // Data Bitwidth
    parameter   GBUS_ADDR   = 12;               // Memory Space
    // 2. Core Memory (WMEM and KV Cache)
    parameter   WMEM_DEPTH  = 1024;             // WMEM Size
    parameter   CACHE_DEPTH = 1024;             // KV Cache Size
    // 3. Core Buffer (LBUF and ABUF)
    parameter   LBUF_DATA   = 8*64;             // LBUF Data Bitwidth
    parameter   LBUF_DEPTH  = 16;               // LBUF Size
    // 4. Computing Logic
    parameter   MAC_NUM   = 64;                 // MAC Line Size
    parameter   IDATA_BIT = 8;                  // Input and Output Bitwidth
    parameter   ODATA_BIT = 32;                 // Partial Sum Bitwidth
    // 5. Config Signals
    parameter   CDATA_BIT = 8;

    // Global Signals
    logic                       clk;
    logic                       rst;
    // Global Config Signals
    logic       [CDATA_BIT-1:0] cfg_acc_num;
    logic       [ODATA_BIT-1:0] cfg_quant_scale;
    logic       [ODATA_BIT-1:0] cfg_quant_bias;
    logic       [ODATA_BIT-1:0] cfg_quant_shift;
    // Channel - Global Bus to Access Core Memory and MAC Result
    // 1. Write Channel
    //      1.1 Chip Interface -> WMEM for Weight Upload
    //      1.2 Chip Interface -> KV Cache for KV Upload (Just Run Attention Test)
    //      1.3 Vector Engine  -> KV Cache for KV Upload (Run Projection and/or Attention)
    // 2. Read Channel
    //      2.1 WMEM       -> Chip Interface for Weight Check
    //      2.2 KV Cache   -> Chip Interface for KV Checnk
    //      2.3 MAC Result -> Vector Engine  for Post Processing
    logic       [GBUS_ADDR-1:0] gbus_addr;
    logic                       gbus_wen;
    logic       [GBUS_DATA-1:0] gbus_wdata;
    logic                       gbus_ren;
    logic [GBUS_DATA-1:0] gbus_rdata;     // To Chip Interface (Debugging) and Vector Engine (MAC)
    logic                 gbus_rvalid;
    // Channel - Core-to-Core Link
    // Vertical for Weight and Key/Value Propagation
    logic                       vlink_enable;
    logic       [GBUS_DATA-1:0] vlink_wdata;
    logic                       vlink_wen;
    logic      [GBUS_DATA-1:0] vlink_rdata;
    logic                      vlink_rvalid;
    // Horizontal for Activation Propagation
    // input                    vlink_enable;   // No HLING_ENABLE for Activaton
    logic       [GBUS_DATA-1:0] hlink_wdata;
    logic                       hlink_wen;
    logic      [GBUS_DATA-1:0] hlink_rdata;
    logic                      hlink_rvalid;
    // Channel - MAC Operation
    // Core Memory Access for Weight and KV Cache
    logic       [GBUS_ADDR-1:0] cmem_waddr;     // Write Value to KV Cache
    logic                       cmem_wen;
    logic       [GBUS_ADDR-1:0] cmem_raddr;
    logic                       cmem_ren;
    // Local Buffer Access for Weight and KV Cache
    //input                     lbuf_mux;       // Annotate for Double-Buffering LBUF
    logic       [LBUF_ADDR-1:0] lbuf_waddr;
    logic       [LBUF_ADDR-1:0] lbuf_raddr;
    logic                       lbuf_ren;
    // Local Buffer Access for Activation
    //input                     abuf_mux;
    logic       [LBUF_ADDR-1:0] abuf_waddr;
    logic       [LBUF_ADDR-1:0] abuf_raddr;
    logic                       abuf_ren;

    core_top #(GBUS_DATA,GBUS_ADDR,WMEM_DEPTH,CACHE_DEPTH,LBUF_DATA,LBUF_DEPTH,MAC_NUM,IDATA_BIT,ODATA_BIT,CDATA_BIT) dut (.*);

endmodule
