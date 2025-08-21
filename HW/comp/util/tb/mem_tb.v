module tb_mem;

    parameter DATA_BIT = 64;
    parameter DEPTH = 1024;
    parameter ADDR_BIT = $clog2(DEPTH);

    reg clk;
    reg rst;

    // SP
    reg [ADDR_BIT-1:0] sp_addr;
    reg sp_wen;
    reg [DATA_BIT-1:0] sp_wdata;
    reg sp_ren;
    wire [DATA_BIT-1:0] sp_rdata;

    // DP
    reg [ADDR_BIT-1:0] dp_waddr;
    reg dp_wen;
    reg [DATA_BIT-1:0] dp_wdata;
    reg [ADDR_BIT-1:0] dp_raddr;
    reg dp_ren;
    wire [DATA_BIT-1:0] dp_rdata;

    // DB
    reg db_sw;
    reg [ADDR_BIT-1:0] db_waddr;
    reg db_wen;
    reg [DATA_BIT-1:0] db_wdata;
    reg [ADDR_BIT-1:0] db_raddr;
    reg db_ren;
    wire [DATA_BIT-1:0] db_rdata;

    mem_sp #(.DATA_BIT(DATA_BIT), .DEPTH(DEPTH)) sp_inst (
        .clk(clk),
        .addr(sp_addr),
        .wen(sp_wen),
        .wdata(sp_wdata),
        .ren(sp_ren),
        .rdata(sp_rdata)
    );

    mem_dp #(.DATA_BIT(DATA_BIT), .DEPTH(DEPTH)) dp_inst (
        .clk(clk),
        .waddr(dp_waddr),
        .wen(dp_wen),
        .wdata(dp_wdata),
        .raddr(dp_raddr),
        .ren(dp_ren),
        .rdata(dp_rdata)
    );

    mem_db #(.DATA_BIT(DATA_BIT), .DEPTH(DEPTH)) db_inst (
        .clk(clk),
        .sw(db_sw),
        .waddr(db_waddr),
        .wen(db_wen),
        .wdata(db_wdata),
        .raddr(db_raddr),
        .ren(db_ren),
        .rdata(db_rdata)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst = 1;
        sp_addr = 0;
        sp_wen = 0;
        sp_wdata = 0;
        sp_ren = 0;

        dp_waddr = 0;
        dp_wen = 0;
        dp_wdata = 0;
        dp_raddr = 0;
        dp_ren = 0;

        db_sw = 0;
        db_waddr = 0;
        db_wen = 0;
        db_wdata = 0;
        db_raddr = 0;
        db_ren = 0;

        #10 rst = 0;

        // SP
        $display("Testing SP RAM...");
        sp_addr = 10;
        sp_wdata = 64'hABCDEF12;
        sp_wen = 1;
        #10;
        sp_wen = 0;
        sp_ren = 1;
        #10;
        $display("SP RAM Read Data: %h", sp_rdata);
        sp_ren = 0;

        // DP
        $display("Testing DP RAM...");
        dp_waddr = 20;
        dp_wdata = 64'hABCDEF12;
        dp_wen = 1;
        #10;
        dp_wen = 0;
        dp_raddr = 20;
        dp_ren = 1;
        #10;
        $display("DP RAM Read Data: %h", dp_rdata);
        dp_ren = 0;

      	// DB(with switch)
      	$display("Testing DB RAM...");
      	
      	db_waddr = 40;
        db_wdata = 64'h87654321;
        db_wen = 1;
        db_sw = 1;
      	$display("Writing 87654321 to bank 0");
        #10;
        db_wen = 0;
        db_raddr = 40;
        db_ren = 1;
      	db_sw = 0;
        #10;
      	$display("DB RAM Read Data (Bank 0): %h", db_rdata);
        db_ren = 0;

        db_waddr = 30;
        db_wdata = 64'h12345678;
        db_wen = 1;
        db_sw = 0;
      	$display("Writing 12345678 to bank 1");
        #10;
        db_wen = 0;
        db_raddr = 30;
        db_ren = 1;
      	db_sw = 1;
        #10;
      	$display("DB RAM Read Data (Bank 1): %h", db_rdata);
        db_ren = 0;

        $stop;
    end

endmodule
