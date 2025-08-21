module SPI_Slave_TOP
#(parameter SPI_MODE = 3)
(
// Control/Data Signals,
input            i_resetn,    // FPGA Reset, active low
input            i_clk,      // FPGA Clock

output logic [`ARR_CONSLUT_ADDR:0] cons_lut_waddr,
output logic cons_lut_wen,
output logic [`ARR_CONSLUT_DATA-1:0] cons_lut_wdata,

// output logic                  ln_lut_wen,
// output logic [$clog2(`SEQ_LENGTH)-1:0] ln_lut_addr,
// output logic [2*`LN_FP_W-1:0] ln_lut_wdata,
// output logic                  ln_lut_ren,
// input        [2*`LN_FP_W-1:0] ln_lut_rdata,
// input						  ln_lut_rvalid,

output logic [$clog2(`INST_REG_DEPTH)-1:0] inst_reg_addr,
output logic inst_reg_wen,
output logic inst_reg_ren,
input  GPT_COMMAND  inst_reg_rdata,
output GPT_COMMAND  inst_reg_wdata,

output logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0] spi_sram_addr,
output logic [`ARR_GBUS_DATA-1:0] spi_sram_wdata,
input        [`ARR_GBUS_DATA-1:0] spi_sram_rdata,
output logic spi_sram_wen,
output logic spi_sram_ren,

output logic [$clog2(`GLOBAL_SRAM_DEPTH)-1:0] spi_sram0_addr,
output logic [`ARR_GBUS_DATA-1:0] spi_sram0_wdata,
input        [`ARR_GBUS_DATA-1:0] spi_sram0_rdata,
output logic spi_sram0_wen,
output logic spi_sram0_ren,

output CFG_ARR_PACKET                      spi_cfg,
output logic                               spi_cfg_en,
output logic [$clog2(`INST_REG_DEPTH)-1:0] spi_cfg_addr,

output logic spi_start,

output logic [`ARR_GBUS_ADDR-1:0]                spi_gbus_addr,
input    	 [`ARR_HNUM-1:0][`ARR_GBUS_DATA-1:0] spi_gbus_rdata,
input  CTRL  									 spi_gbus_rvalid,
output CTRL                                      spi_gbus_ren,
output logic [`ARR_GBUS_DATA-1:0]                spi_gbus_wdata,
output CTRL                                      spi_gbus_wen,

//clk cfg
output  logic pll_cfg_vld,
output  logic [16-1:0][8-1:0] pll_cfg, //TODO:

// SPI Interface
input        i_SPI_Clk,
output logic o_SPI_MISO,
input        i_SPI_MOSI,
input        i_SPI_CS_n        // active low
);

logic rx_data_valid;
logic [7:0] rx_data_byte;
logic tx_data_valid;
logic [7:0] tx_data_byte;
logic rx_cmd_valid;

SPI_WRWRD_FSM				spi_wrwrd_state;
logic	[4-1:0]				spi_wrwrd_bytecnt;

SPI_RDWRD_FSM				spi_rdwrd_state;
logic	[4-1:0]				spi_rdwrd_bytecnt;
logic	[4-1:0]				spi_rdwrd_addrcnt;

SPI_SLAVE # (
	.SPI_MODE(SPI_MODE)
) u_SPI_SLAVE (
	.i_Rst_L		(i_resetn),
	.i_Clk			(i_clk),
	.o_RX_DV		(rx_data_valid),
	.o_RX_CMDV		(rx_cmd_valid),
	.o_RX_Byte		(rx_data_byte),
	.i_TX_DV		(tx_data_valid),
	.i_TX_Byte		(tx_data_byte),

	.i_SPI_Clk		(i_SPI_Clk),
	.o_SPI_MISO		(o_SPI_MISO),
	.i_SPI_MOSI		(i_SPI_MOSI),
	.i_SPI_CS_n		(i_SPI_CS_n)
);

logic [8-1:0]  		cons_lut_waddr_reg;
logic [2*8-1:0]   	cons_lut_wdata_reg;
logic 		    	cons_lut_wen_reg;
logic [3*8-1:0] 	cons_lut_w_reg;
logic				cons_lut_wvalid;

logic [8-1:0] 		inst_waddr_reg;
logic [8-1:0] 		inst_wdata_reg;
logic 				inst_wen_reg;
logic [16-1:0] 		inst_w_reg;
logic		   		inst_wvalid;

logic [2*8-1:0] 	sram_waddr_reg;
logic [8*8-1:0] 	sram_wdata_reg;
logic 				sram_wen_reg;
logic [10*8-1:0]  	sram_w_reg;
logic		   		sram_wvalid;

logic [8-1:0] 		cfg_waddr_reg;
logic [7*8-1:0]	 	cfg_wdata_reg;
logic 			   	cfg_wen_reg;
logic [8*8-1:0]  	cfg_w_reg;
logic		   		cfg_wvalid;

logic         start_wvalid;
logic		  start_reg;

logic [3*8-1:0] 	gbus_waddr_reg;
logic [8*8-1:0]	 	gbus_wdata_reg;
logic 			   	gbus_wen_reg;
logic [11*8-1:0]  	gbus_w_reg;
logic		   		gbus_wvalid;

logic [16*8-1:0] 	pll_cfg_wdata_reg;
logic 				pll_cfg_wen_reg;
logic [16*8-1:0]  	pll_cfg_w_reg;
logic		   		pll_cfg_wvalid;

always_ff@(posedge i_clk or negedge i_resetn) begin
	if (!i_resetn) begin
		spi_wrwrd_state	<= SPI_WRWRD_IDLE;
		spi_wrwrd_bytecnt	<= 'd0;
		
		cons_lut_wdata_reg <= 'd0;
		cons_lut_waddr_reg <= 'd0;
		cons_lut_wen_reg   <= 'd0;
		cons_lut_w_reg <= 'd0;
		cons_lut_wvalid <= '0;

		inst_waddr_reg <= 'd0;
		inst_wdata_reg<= 'd0;
		inst_wen_reg<= 'd0;
		inst_w_reg<= 'd0;
		inst_wvalid<= 'd0;

		start_wvalid <= 'd0;
		start_reg <= 'd0;
		
		sram_waddr_reg <= 'd0;
		sram_wdata_reg <= 'd0;
		sram_wen_reg   <= 'd0;
		sram_w_reg     <= 'd0;
		sram_wvalid    <= 'd0;
		
		cfg_waddr_reg <= 'd0;
		cfg_wdata_reg <= 'd0;
		cfg_wen_reg   <= 'd0;
		cfg_w_reg     <= 'd0;
		cfg_wvalid    <= 'd0;
		
		gbus_waddr_reg <= 'd0;
		gbus_wdata_reg <= 'd0;
		gbus_wen_reg   <= 'd0;
		gbus_w_reg     <= 'd0;
		gbus_wvalid    <= 'd0;

		pll_cfg_wdata_reg <= 'd0;
		pll_cfg_wen_reg   <= 'd0;
		pll_cfg_w_reg     <= 'd0;
		pll_cfg_wvalid    <= 'd0;

	end
	else begin
		case(spi_wrwrd_state)
			SPI_WRWRD_IDLE: begin
				cons_lut_wdata_reg <= 'd0;
				cons_lut_waddr_reg <= 'd0;
				cons_lut_wen_reg   <= 'd0;
				cons_lut_w_reg	<= 'd0;
				cons_lut_wvalid <= '0;

				inst_waddr_reg <= 'd0;
				inst_wdata_reg<= 'd0;
				inst_wen_reg<= 'd0;
				inst_w_reg<= 'd0;
				inst_wvalid<= 'd0;

				start_wvalid <= 'd0;
				start_reg <= 'd0;
				
				sram_waddr_reg <= 'd0;
				sram_wdata_reg <= 'd0;
				sram_wen_reg   <= 'd0;
				sram_w_reg     <= 'd0;
				sram_wvalid    <= 'd0;
				
				cfg_waddr_reg <= 'd0;
				cfg_wdata_reg <= 'd0;
				cfg_wen_reg   <= 'd0;
				cfg_w_reg     <= 'd0;
				cfg_wvalid    <= 'd0;
				
				gbus_waddr_reg <= 'd0;
				gbus_wdata_reg <= 'd0;
				gbus_wen_reg   <= 'd0;
				gbus_w_reg     <= 'd0;
				gbus_wvalid    <= 'd0;

				pll_cfg_wdata_reg <= 'd0;
				pll_cfg_wen_reg   <= 'd0;
				pll_cfg_w_reg     <= 'd0;
				pll_cfg_wvalid    <= 'd0;

				if (rx_cmd_valid && rx_data_byte == SPI_CONS_WR_WORD) begin
					spi_wrwrd_state	<= SPI_WRWRD_GET_CONS;
				end else if(rx_cmd_valid && rx_data_byte == SPI_INST_WR_WORD) begin
					spi_wrwrd_state <= SPI_WRWRD_GET_INST;
				end else if (rx_cmd_valid && rx_data_byte == SPI_SRAM_WR_WORD) begin
                    spi_wrwrd_state <= SPI_WRWRD_GET_SRAM;
                end else if(rx_cmd_valid && rx_data_byte == SPI_CFG_WR_WORD) begin
                    spi_wrwrd_state <= SPI_WRWRD_GET_CFG;
                end else if(rx_cmd_valid && rx_data_byte == SPI_START_WR_WORD) begin
                    spi_wrwrd_state <= SPI_WRWRD_GET_START;
                end else if(rx_cmd_valid && rx_data_byte == SPI_GBUS_WR_WORD) begin
                    spi_wrwrd_state <= SPI_WRWRD_GET_GBUS;
                end else if(rx_cmd_valid && rx_data_byte == SPI_PLL_CFG_WR_WORD) begin
                    spi_wrwrd_state <= SPI_WRWRD_GET_PLL_CFG;
                end

			end
			SPI_WRWRD_GET_CONS: begin
				if (rx_data_valid) begin
					cons_lut_w_reg <= {cons_lut_w_reg[15:0],rx_data_byte};
					if (spi_wrwrd_bytecnt == 'd2) begin
						spi_wrwrd_bytecnt	<= 4'd0;
						spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						cons_lut_wvalid  <= 1'b1;
					end
					else begin
						spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 4'd1;
					end
				end
			end
			SPI_WRWRD_GET_INST: begin
				if (rx_data_valid) begin
					inst_w_reg <= {inst_w_reg[7:0],rx_data_byte};
					if (spi_wrwrd_bytecnt == 'd1) begin
						spi_wrwrd_bytecnt	<= 4'd0;
						spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						inst_wvalid  <= 1'b1;
					end
					else begin
						spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 4'd1;
					end
				end
			end
			SPI_WRWRD_GET_SRAM: begin
                if (rx_data_valid) begin
					sram_w_reg <= {sram_w_reg[71:0],rx_data_byte};
					if (spi_wrwrd_bytecnt == 'd9) begin
						spi_wrwrd_bytecnt	<= 4'd0;
						spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						sram_wvalid  <= 1'b1;
					end
					else begin
						spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 4'd1;
					end
				end
            end
			SPI_WRWRD_GET_PLL_CFG: begin
                if (rx_data_valid) begin
					pll_cfg_w_reg <= {pll_cfg_w_reg[119:0],rx_data_byte};
					if (spi_wrwrd_bytecnt == 'd15) begin
						spi_wrwrd_bytecnt	<= 4'd0;
						spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						pll_cfg_wvalid  <= 1'b1;
					end
					else begin
						spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 4'd1;
					end
				end
            end
			SPI_WRWRD_GET_CFG: begin
                if (rx_data_valid) begin
					cfg_w_reg <= {cfg_w_reg[55:0],rx_data_byte};
					if (spi_wrwrd_bytecnt == 'd7) begin
						spi_wrwrd_bytecnt	<= 4'd0;
						spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						cfg_wvalid  <= 1'b1;
					end
					else begin
						spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 4'd1;
					end
				end
            end
            SPI_WRWRD_GET_START: begin
				start_wvalid<=1'b1;
				spi_wrwrd_state	<= SPI_WRWRD_WRITE;
            end
            SPI_WRWRD_GET_GBUS: begin
                if (rx_data_valid) begin
					gbus_w_reg <= {gbus_w_reg[79:0],rx_data_byte};
					if (spi_wrwrd_bytecnt == 'd10) begin
						spi_wrwrd_bytecnt	<= 4'd0;
						spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						gbus_wvalid  <= 1'b1;
					end
					else begin
						spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 4'd1;
					end
				end
            end
			SPI_WRWRD_WRITE: begin
				spi_wrwrd_state	<= SPI_WRWRD_IDLE;
				
				cons_lut_wdata_reg <= cons_lut_w_reg[15:0];
				cons_lut_waddr_reg <= cons_lut_w_reg[23:16];
				cons_lut_wen_reg <= cons_lut_wvalid;
				
				inst_wdata_reg <= inst_w_reg[7:0];
				inst_waddr_reg <= inst_w_reg[15:8];
				inst_wen_reg <= inst_wvalid;

				sram_wdata_reg <= sram_w_reg[63:0];
				sram_waddr_reg <= sram_w_reg[79:64];
				sram_wen_reg   <= sram_wvalid;

				cfg_wdata_reg <= cfg_w_reg[55:0];
				cfg_waddr_reg <= cfg_w_reg[63:56];
				cfg_wen_reg   <= cfg_wvalid;
				
				start_reg <= start_wvalid;

				gbus_wdata_reg <= gbus_w_reg[63:0];
				gbus_waddr_reg <= gbus_w_reg[87:64];
				gbus_wen_reg   <= gbus_wvalid;

				pll_cfg_wdata_reg <= pll_cfg_w_reg;
				pll_cfg_wen_reg   <= pll_cfg_wvalid;

			end
		endcase
	end
end

logic [8-1:0] inst_raddr_reg;
logic		  inst_ren_reg;

logic [2*8-1:0] sram_raddr_reg;
logic 			sram_ren_reg;
logic [63:0] spi_rdwrd_sram;
logic 			sram_rvalid;
logic 			sram0_rvalid;
logic           inst_rvalid;
logic           gbus_rvalid;

logic [3*8-1:0] gbus_raddr_reg;
logic 			gbus_ren_reg;
logic [63:0] spi_rdwrd_gbus;
logic [31:0] spi_rdwrd_rdata;

always_ff@(posedge i_clk or negedge i_resetn) begin
	if (!i_resetn) begin
		spi_rdwrd_state	<= SPI_RDWRD_IDLE;
		spi_rdwrd_bytecnt	<= 'd0;
		spi_rdwrd_rdata		<= 32'd0;

		spi_rdwrd_addrcnt <= 0;

		tx_data_valid	<= 1'b0;
		tx_data_byte	<= 8'd0;

		inst_raddr_reg <= 'd0;
		inst_ren_reg <= 'd0;

		sram_raddr_reg <= 'd0;
		sram_ren_reg <= 'd0;
		spi_rdwrd_sram <= 'd0;

		gbus_raddr_reg <= 'd0;
		gbus_ren_reg <= 'd0;
		spi_rdwrd_gbus <= 'd0;
	end
	else begin
		case(spi_rdwrd_state)
			SPI_RDWRD_IDLE: begin
				tx_data_valid	<= 1'b0;
				tx_data_byte	<= 8'd0;
				if (rx_cmd_valid) begin
					case(rx_data_byte)
						SPI_INST_RD_WORD: begin
							spi_rdwrd_state	<= SPI_GET_INST_ADDR;
						end
						SPI_SRAM_RD_WORD: begin
							spi_rdwrd_state	<= SPI_GET_SRAM_ADDR;
						end
						SPI_GBUS_RD_WORD: begin
							spi_rdwrd_state	<= SPI_GET_GBUS_ADDR;
						end
					endcase
				end
			end
			SPI_GET_INST_ADDR: begin
				if (rx_data_valid) begin
					inst_raddr_reg <= rx_data_byte;
					inst_ren_reg <= 1'b1;
					spi_rdwrd_state	<= SPI_RDWRD_GET_INST;
				end
			end
			SPI_GET_SRAM_ADDR: begin
				if(rx_data_valid) begin
					sram_raddr_reg <= {sram_raddr_reg[7:0],rx_data_byte};
					if (spi_rdwrd_addrcnt == 'd1) begin
						spi_rdwrd_addrcnt	<= 4'd0;
						spi_rdwrd_state	<= SPI_RDWRD_GET_SRAM;
						sram_ren_reg  <= 1'b1;
					end
					else begin
						spi_rdwrd_addrcnt	<= spi_rdwrd_addrcnt + 4'd1;
					end
				end
			end
			SPI_GET_GBUS_ADDR: begin
				if(rx_data_valid) begin
					gbus_raddr_reg <= {gbus_raddr_reg[15:0],rx_data_byte};
					if (spi_rdwrd_addrcnt == 'd2) begin
						spi_rdwrd_addrcnt	<= 4'd0;
						spi_rdwrd_state	<= SPI_RDWRD_GET_GBUS;
						gbus_ren_reg  <= 1'b1;
					end
					else begin
						spi_rdwrd_addrcnt	<= spi_rdwrd_addrcnt + 4'd1;
					end
				end
			end
			SPI_RDWRD_GET_INST: begin
				inst_ren_reg <= 1'b0;
				if(inst_rvalid) begin
					spi_rdwrd_state	<= SPI_RDWRD_IDLE;
					tx_data_valid	<= 1'b1;
					tx_data_byte	<= inst_reg_rdata;
				end
			end
			SPI_RDWRD_GET_SRAM: begin
				sram_ren_reg  <= 1'b0;
				// spi_rdwrd_state	<= SPI_RDWRD_SRAM; here!
				// tx_data_valid	<= 1'b1; here
				if(sram_rvalid) begin
					spi_rdwrd_state	<= SPI_RDWRD_SRAM;  //here
					tx_data_valid	<= 1'b1;//here
					tx_data_byte	<= spi_sram_rdata[63:56];
					spi_rdwrd_sram	<= (spi_sram_rdata << 'd8);
				end else if(sram0_rvalid) begin
					spi_rdwrd_state	<= SPI_RDWRD_SRAM;//here
					tx_data_valid	<= 1'b1;//here
					tx_data_byte	<= spi_sram0_rdata[63:56];
					spi_rdwrd_sram	<= (spi_sram0_rdata << 'd8);
				end
				
			end
			SPI_RDWRD_GET_GBUS: begin
				gbus_ren_reg <= 1'b0;
				if(gbus_rvalid)begin
					spi_rdwrd_state	<= SPI_RDWRD_GBUS;
					tx_data_valid	<= 1'b1;
					for(int i=0;i<`ARR_HNUM;i++) begin
						if(|spi_gbus_rvalid[i]) begin
							tx_data_byte	<= spi_gbus_rdata[i][63:56];
							spi_rdwrd_gbus	<= (spi_gbus_rdata[i] << 'd8);
						end
					end
				end
			end
			SPI_RDWRD_SRAM: begin				
				if (rx_data_valid) begin
					tx_data_valid	<= 1'b1;
					tx_data_byte	<= spi_rdwrd_sram[63:56];
					spi_rdwrd_sram	<= (spi_rdwrd_sram << 'd8);
					if (spi_rdwrd_bytecnt == 4'd6) begin
						spi_rdwrd_bytecnt	<= 4'd0;
						spi_rdwrd_state		<= SPI_RDWRD_IDLE;
					end
					else begin
						spi_rdwrd_bytecnt	<= spi_rdwrd_bytecnt + 'd1;
					end
				end
				else begin
					tx_data_valid	<= 1'b0;
					tx_data_byte	<= 8'd0;
				end
			end
			SPI_RDWRD_GBUS: begin
				if (rx_data_valid) begin
					tx_data_valid	<= 1'b1;
					tx_data_byte	<= spi_rdwrd_gbus[63:56];
					spi_rdwrd_gbus	<= (spi_rdwrd_gbus << 'd8);
					if (spi_rdwrd_bytecnt == 4'd6) begin
						spi_rdwrd_bytecnt	<= 4'd0;
						spi_rdwrd_state		<= SPI_RDWRD_IDLE;
					end
					else begin
						spi_rdwrd_bytecnt	<= spi_rdwrd_bytecnt + 'd1;
					end
				end
				else begin
					tx_data_valid	<= 1'b0;
					tx_data_byte	<= 8'd0;
				end
			end
		endcase
	end
end

always_comb begin
	cons_lut_waddr 	= cons_lut_waddr_reg;
	cons_lut_wdata 	= cons_lut_wdata_reg; 
	cons_lut_wen   	= cons_lut_wen_reg;

	inst_reg_addr  	= inst_wen_reg ? inst_waddr_reg : inst_raddr_reg;
	inst_reg_wen	= inst_wen_reg;
	inst_reg_ren	= inst_ren_reg;
	inst_reg_wdata	= inst_wdata_reg;

	spi_start		= start_reg;

	pll_cfg_vld     = pll_cfg_wen_reg;
	for(int i = 0;i < 16;i++)begin
		pll_cfg[i]=pll_cfg_wdata_reg[i*8+:8];
	end

	spi_sram_addr = sram_wen_reg ? sram_waddr_reg : sram_raddr_reg;
	if(sram_raddr_reg[12]) begin
		spi_sram_ren = sram_ren_reg;
	end
	else begin
		spi_sram_ren = '0;
	end
	if(sram_waddr_reg[12]) begin
		spi_sram_wdata = sram_wdata_reg;
		spi_sram_wen = sram_wen_reg;
	end
	else begin
		spi_sram_wdata = '0;
		spi_sram_wen = '0;
	end
	
	spi_sram0_addr = sram_wen_reg ? sram_waddr_reg : sram_raddr_reg;
	if(~sram_raddr_reg[12]) begin
		spi_sram0_ren = sram_ren_reg;
	end
	else begin
		spi_sram0_ren = '0;
	end
	if(~sram_waddr_reg[12]) begin
		spi_sram0_wdata = sram_wdata_reg;
		spi_sram0_wen = sram_wen_reg;
	end
	else begin
		spi_sram0_wdata = '0;
		spi_sram0_wen = '0;
	end

	spi_cfg = cfg_wdata_reg;
	spi_cfg_en = cfg_wen_reg;
	spi_cfg_addr = cfg_waddr_reg;

	spi_gbus_ren = '0;
	spi_gbus_wen = '0;
	spi_gbus_addr = gbus_wen_reg ? gbus_waddr_reg : gbus_raddr_reg;
	spi_gbus_ren[gbus_raddr_reg[17:15]][gbus_raddr_reg[14:12]] = gbus_ren_reg;
	spi_gbus_wen[gbus_waddr_reg[17:15]][gbus_waddr_reg[14:12]] = gbus_wen_reg;
	spi_gbus_wdata = gbus_wdata_reg;
end
always_ff@(posedge i_clk or negedge i_resetn) begin
	if(~i_resetn) begin
		sram0_rvalid<=1'b0;
		sram_rvalid<=1'b0;
		inst_rvalid<=1'b0;
		gbus_rvalid<=1'b0;
	end
	else begin
		sram0_rvalid<=spi_sram0_ren;
		sram_rvalid<=spi_sram_ren;
		inst_rvalid<=inst_reg_ren;
		gbus_rvalid<=gbus_ren_reg;
	end
end
endmodule
