module SPI_DMA_M33 (
	input				i_clk,
	input				i_resetn,

	input				i_rx_data_valid,
	input				i_rx_cmd_valid,
	input	[8-1:0]		i_rx_data_byte,
	output	logic				o_tx_data_rdy,
	output	logic	[32-1:0]	o_tx_data_word,

	output	logic				o_spi_m33_softrstn,
	output	SRAM_CONFIG_PKT		o_spi_m33sram_config,	
	output	logic				o_spi_m33cram_ctrl,
 	output	logic				o_spi_m33cram_cs,
	output	logic	[4-1:0]		o_spi_m33cram_we,
	output	logic	[32-1:0]	o_spi_m33cram_addr,
	output	logic	[32-1:0]	o_spi_m33cram_wdata,
	input			[32-1:0]	i_spi_m33cram_rdata
);

// Signal Declarations
	logic	[8-1:0]		spi_cmd_latched;
	SPI_M33DMA_FSM		spi_m33dma_state;
	logic	[3	-1:0]	spi_m33dma_bytecnt;	
	logic	[32-1:0]	spi_m33sram_config;

	assign o_spi_m33sram_config.stov 	= spi_m33sram_config[28];
	assign o_spi_m33sram_config.ema 	= spi_m33sram_config[26:24];
	assign o_spi_m33sram_config.emaw 	= spi_m33sram_config[21:20];
	assign o_spi_m33sram_config.emas 	= spi_m33sram_config[16];
	assign o_spi_m33sram_config.ret1n 	= spi_m33sram_config[12];
	assign o_spi_m33sram_config.ret2n	= spi_m33sram_config[8];
	assign o_spi_m33sram_config.rawl 	= 1'b1;
	assign o_spi_m33sram_config.rawlm 	= spi_m33sram_config[5:4];
	assign o_spi_m33sram_config.wabl 	= 1'b1;
	assign o_spi_m33sram_config.wablm 	= spi_m33sram_config[2:0];

// SPI DMA M33 FSM
	always_ff@(posedge i_clk or negedge i_resetn) begin
		if (!i_resetn) begin
			spi_cmd_latched	<= SPI_NOP;
			spi_m33dma_state	<= SPI_M33DMA_IDLE;

			o_tx_data_rdy	<= 1'b0;
			o_tx_data_word	<= 32'd0;

			o_spi_m33_softrstn	<= 1'b0;
			o_spi_m33cram_ctrl	<= 1'b0;
			o_spi_m33cram_cs	<= 1'b0;
			o_spi_m33cram_we	<= 4'b0;
			o_spi_m33cram_addr	<= 32'd0;
			o_spi_m33cram_wdata	<= 32'd0;			

			spi_m33sram_config	<= {
									3'd0, 1'b0, // STOV
									1'b0, 3'b111, // EMA
									2'd0, 2'b11, // EMAW
									3'd0, 1'b1, // EMAS
									3'd0, 1'b1, // RET1N
									3'd0, 1'b1, // RET2N
									2'd0, 2'b00, // RAWLM
									1'b0, 3'b001 // WABLM
								   };

			spi_m33dma_bytecnt	<= 3'd0;
		end
		else begin
			case(spi_m33dma_state)
				SPI_M33DMA_IDLE: begin
					if (i_rx_cmd_valid) begin
						spi_cmd_latched	<= i_rx_data_byte;
						case(i_rx_data_byte)
							SPI_M33_RSTN: spi_m33dma_state	<= SPI_M33DMA_RSTN;
							SPI_M33_CRAMCTRL: spi_m33dma_state	<= SPI_M33DMA_CRAMCTRL;
							SPI_M33_SRAMCFG: spi_m33dma_state	<= SPI_M33DMA_SRAMCFG;
							SPI_M33_WRWORD: spi_m33dma_state	<= SPI_M33DMA_GETADDR;
							SPI_M33_LDWORD: spi_m33dma_state	<= SPI_M33DMA_GETADDR;
						endcase

						o_tx_data_rdy	<= 1'b0;
						o_tx_data_word	<= 32'd0;
						o_spi_m33cram_cs	<= 1'b0;
						o_spi_m33cram_we	<= 4'b0;
						o_spi_m33cram_addr	<= 32'd0;
						o_spi_m33cram_wdata	<= 32'd0;			
						spi_m33dma_bytecnt	<= 3'd0;
					end
					else begin
						spi_cmd_latched	<= SPI_NOP;
						o_tx_data_rdy	<= 1'b0;
						o_tx_data_word	<= 32'd0;
						o_spi_m33cram_cs	<= 1'b0;
						o_spi_m33cram_we	<= 4'b0;
						o_spi_m33cram_addr	<= 32'd0;
						o_spi_m33cram_wdata	<= 32'd0;			
						spi_m33dma_bytecnt	<= 3'd0;
					end
				end
				SPI_M33DMA_RSTN: begin
					if (i_rx_data_valid) begin
						o_spi_m33_softrstn	<= i_rx_data_byte[0];
						spi_m33dma_state	<= SPI_M33DMA_IDLE;
					end
				end
				SPI_M33DMA_CRAMCTRL: begin	
					if (i_rx_data_valid) begin
						o_spi_m33cram_ctrl	<= i_rx_data_byte[0];
						spi_m33dma_state	<= SPI_M33DMA_IDLE;
					end
				end
				SPI_M33DMA_SRAMCFG: begin
					if (i_rx_data_valid) begin
						spi_m33sram_config	<= {spi_m33sram_config[23:0], i_rx_data_byte};
						if (spi_m33dma_bytecnt == 3'd3) begin
							spi_m33dma_bytecnt	<= 3'd0;
							spi_m33dma_state	<= SPI_M33DMA_IDLE;
						end
						else begin
							spi_m33dma_bytecnt	<= spi_m33dma_bytecnt + 3'd1;
						end
					end
				end
				SPI_M33DMA_GETADDR: begin
					if (i_rx_data_valid) begin
						o_spi_m33cram_addr	<= {o_spi_m33cram_addr[23:0], i_rx_data_byte};
						if (spi_m33dma_bytecnt == 3'd3) begin
							spi_m33dma_bytecnt	<= 3'd0;
							case(spi_cmd_latched)
								SPI_M33_WRWORD: begin
									spi_m33dma_state	<= SPI_M33DMA_GETWDATA;
								end
								SPI_M33_LDWORD: begin
									spi_m33dma_state	<= SPI_M33DMA_LOAD1;
									o_spi_m33cram_cs	<= 1'b1;
									o_spi_m33cram_we	<= 4'b0;
								end
							endcase
						end
						else begin
							spi_m33dma_bytecnt	<= spi_m33dma_bytecnt + 3'd1;
						end
					end
				end
				SPI_M33DMA_GETWDATA: begin
					if (i_rx_data_valid) begin
						o_spi_m33cram_wdata	<= {o_spi_m33cram_wdata[23:0], i_rx_data_byte};
						if (spi_m33dma_bytecnt == 3'd3) begin
							spi_m33dma_bytecnt	<= 3'd0;
							spi_m33dma_state	<= SPI_M33DMA_WRITE;
							o_spi_m33cram_cs	<= 1'b1;
							o_spi_m33cram_we	<= 4'b1111;
						end
						else begin
							spi_m33dma_bytecnt	<= spi_m33dma_bytecnt + 3'd1;
						end
					end
				end
				SPI_M33DMA_WRITE: begin
					spi_m33dma_state	<= SPI_M33DMA_IDLE;
					o_spi_m33cram_cs	<= 1'b0;
					o_spi_m33cram_we	<= 4'b0;
				end
				SPI_M33DMA_LOAD1: begin
					spi_m33dma_state	<= SPI_M33DMA_LOAD2;
					o_spi_m33cram_cs	<= 1'b0;
					o_spi_m33cram_we	<= 4'b0;
				end
				SPI_M33DMA_LOAD2: begin
					spi_m33dma_state	<= SPI_M33DMA_IDLE;
					o_spi_m33cram_cs	<= 1'b0;
					o_tx_data_rdy		<= 1'b1;
					o_tx_data_word		<= i_spi_m33cram_rdata;
				end
			endcase
		end
	end
endmodule

module SPI_TOP # (
	parameter SPI_MODE = 0,
	parameter BASEADDR = 32'h0
)
(
	// Chip Clk and Reset
	input						i_clk,
	input						i_resetn,
	
	// SPI Interface to off-chip host
	input 		 		  		SPICLK, 			    // SPI Clock
   	output 	logic		 		SPIMISO, 			// SPI Master-in Slave-out
   	input			   			SPIMOSI,				// SPI Master-out Slave-in
   	input 				   		SPICSn,				// SPI Chip Select. Low-active
   	output 	logic		   		SPIIRQ, 				// Interrupt from M33 to SPI Master

	// AHB Interface to M33
	AHB_IF.SLAVE				spi_ahbs_if,
	output	logic				o_spi_m33_irq,

	// DMA to Cortex-M33
	output	logic				o_spi_m33_softrstn,	
	output	SRAM_CONFIG_PKT		o_spi_m33sram_config,
	output	logic				o_spi_m33cram_ctrl,
 	output	logic				o_spi_m33cram_cs,
	output	logic	[4-1:0]		o_spi_m33cram_we,
	output	logic	[32-1:0]	o_spi_m33cram_addr,
	output	logic	[32-1:0]	o_spi_m33cram_wdata,
	input			[32-1:0]	i_spi_m33cram_rdata,

	// Output ports shared by all other built-in-block SPI DMA interfaces (NVPU, MRAM, OGM, etc.)
	output	logic				o_rx_data_valid,
	output	logic				o_rx_cmd_valid,
	output	logic	[8-1:0]		o_rx_data_byte,

	// DMA with NVPU
	input						i_tx_data_rdy_nvpu,
	input			[32-1:0]	i_tx_data_word_nvpu,

	// DMA with MRAM
	input						i_tx_data_rdy_mram,
	input			[32-1:0]	i_tx_data_word_mram,
	input						i_spi_mram_busy,

	// DMA with OGM 
	input						i_tx_data_rdy_ogm,
	input			[32-1:0]	i_tx_data_word_ogm
);

// Local Parameters
	// TX and RX FIFOs
	localparam	WRD_SIZE	= 'd32;
	localparam	WR_WIDTH	= 'd32;
	localparam	RD_WIDTH	= 'd32;
   	localparam 	WNWRD_WIDTH	= $clog2((WR_WIDTH/WRD_SIZE)+1); // Bus width for write-#words
   	localparam 	RNWRD_WIDTH	= $clog2((RD_WIDTH/WRD_SIZE)+1); // Bus width for read-#words
   	localparam 	DEPTH_BIT  	= WRD_SIZE * 16; // FIFO Depth in unit of bit
   	localparam 	PTR_WIDTH	= $clog2((DEPTH_BIT/WRD_SIZE)+1); // Bus width for fifo pointer

// Signal Declarations
  // SPI_SLAVE Signals
	logic						rx_data_valid;
	logic						rx_cmd_valid;
	logic	[8-1:0]				rx_data_byte;
	logic						tx_data_valid;
	logic	[8-1:0]				tx_data_byte;

  // Normal FIFO R/W CMDs-related Signals
	SPI_WRWRD_FSM				spi_wrwrd_state;
	logic	[3-1:0]				spi_wrwrd_bytecnt;

	SPI_RDWRD_FSM				spi_rdwrd_state;
	logic	[3-1:0]				spi_rdwrd_bytecnt;
	logic	[32-1:0]			spi_rdwrd_rdata;

  // Interrupt Signals
	logic						spis_rst_irq_toext;

	logic	[32-1:0]			spis_irq_code_toext;

	logic						spis_irq_frmext;
	logic						spis_rst_irq_frmext;
	
	logic	[32-1:0]			spis_irq_code_frmext;
	SPI_ICFE_FSM				spi_icfe_state;
	logic	[3-1:0]				spi_icfe_bytecnt;
  
  // TX and RX FIFO Signals
	// TX FIFO Signals
	logic						spis_txfifo_wreq;
	logic	[WNWRD_WIDTH-1:0]	spis_txfifo_wnwrd;
	logic	[WR_WIDTH-1:0]		spis_txfifo_wdata;
	logic						spis_txfifo_wrdy;

	logic						spis_txfifo_rreq;
	logic	[RNWRD_WIDTH-1:0]	spis_txfifo_rnwrd;
	logic	[RD_WIDTH-1:0]		spis_txfifo_rdata;
	logic						spis_txfifo_rrdy;			

	logic	[PTR_WIDTH-1:0]		spis_txfifo_datacnt;	
	logic						spis_txfifo_full;
	logic						spis_txfifo_empty;

	// RX FIFO Signals
	logic						spis_rxfifo_wreq;
	logic	[WNWRD_WIDTH-1:0]	spis_rxfifo_wnwrd;
	logic	[WR_WIDTH-1:0]		spis_rxfifo_wdata;
	logic						spis_rxfifo_wrdy;

	logic						spis_rxfifo_rreq;
	logic	[5-1:0]				spis_rxfifo_rnwrd;
	logic	[RD_WIDTH-1:0]		spis_rxfifo_rdata;
	logic						spis_rxfifo_rrdy;			

	logic	[PTR_WIDTH-1:0]		spis_rxfifo_datacnt;	
	logic						spis_rxfifo_full;
	logic						spis_rxfifo_empty;

  // SPI-AHB Interface Signals
	// AHB Bridge Signals
	logic						valid_m2s;
	logic	[32-1:0]			haddr;
	logic	[2	:0]				hsize;
	logic						hwrite;
	logic	[32-1:0]			hwdata;
	logic						hreadyout;
	logic	[32-1:0]			hrdata;
	logic						hresp;

	// SFRMAP Signals
	logic						ahbs_spis_txfifo_wreq;
	logic	[5-1:0]				ahbs_spis_txfifo_wnwrd;
	logic	[32-1:0]			ahbs_spis_txfifo_wdata;

  // SPI DMA Signals
	// SPI_DMA_M33 Signals
	logic						tx_data_rdy_m33;
	logic	[32-1:0]			tx_data_word_m33;

// Module Instantiations
	// SPI_SLAVE
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

		.i_SPI_Clk		(SPICLK),
		.o_SPI_MISO		(SPIMISO),
		.i_SPI_MOSI		(SPIMOSI),
		.i_SPI_CS_n		(SPICSn)
	);
	assign o_rx_data_valid	= rx_data_valid;
	assign o_rx_cmd_valid	= rx_cmd_valid;
	assign o_rx_data_byte	= rx_data_byte;

  // SPI FIFOs
	// SPI TX FIFO
	SYNCFIFO # (
		.WRD_SIZE		(WRD_SIZE),
		.WR_WIDTH		(WR_WIDTH),
		.RD_WIDTH		(RD_WIDTH),
		.DEPTH_BIT		(DEPTH_BIT)
	) u_SPI_TXFIFO (
		.i_clk				(i_clk),
		.i_resetn			(i_resetn),

		.i_wreq				(spis_txfifo_wreq),
		.i_wnwrd			(spis_txfifo_wnwrd),
		.i_wdata			(spis_txfifo_wdata),
		.o_wrdy				(spis_txfifo_wrdy),

		.i_rreq				(spis_txfifo_rreq),
		.i_rnwrd			(spis_txfifo_rnwrd),
		.o_rrdy				(spis_txfifo_rrdy),
		.o_rdata			(spis_txfifo_rdata),
		.o_datacnt			(spis_txfifo_datacnt)
	);
	assign	spis_txfifo_full	= (spis_txfifo_datacnt == 'd16);
	assign	spis_txfifo_empty	= (spis_txfifo_datacnt == 'd0);
	always_comb begin // TX FIFO Write Port Arbitration
		spis_txfifo_wreq	= 1'b0;
		spis_txfifo_wnwrd	= 'd0;
		spis_txfifo_wdata	= 32'd0;

		if (tx_data_rdy_m33) begin // SPI DMA Cortex-M33
			spis_txfifo_wreq	= 1'b1;
			spis_txfifo_wnwrd	= 'd1;
			spis_txfifo_wdata	= tx_data_word_m33;
		end
		else if (i_tx_data_rdy_nvpu) begin // SPI DMA NVPU
			spis_txfifo_wreq	= 1'b1;
			spis_txfifo_wnwrd	= 'd1;
			spis_txfifo_wdata	= i_tx_data_word_nvpu;
		end
		else if (i_tx_data_rdy_mram) begin // SPI DMA MRAM
			spis_txfifo_wreq	= 1'b1;
			spis_txfifo_wnwrd	= 'd1;
			spis_txfifo_wdata	= i_tx_data_word_mram;
		end
		else if (i_tx_data_rdy_ogm) begin // SPI DMA OGM
			spis_txfifo_wreq	= 1'b1;
			spis_txfifo_wnwrd	= 'd1;
			spis_txfifo_wdata	= i_tx_data_word_ogm;
		end
		else if (ahbs_spis_txfifo_wreq) begin // M33 Write FIFO through AHBS
			spis_txfifo_wreq	= 1'b1;
			spis_txfifo_wnwrd	= ahbs_spis_txfifo_wnwrd;
			spis_txfifo_wdata	= ahbs_spis_txfifo_wdata;
		end
	end

	// SPI RX FIFO
	SYNCFIFO # (
		.WRD_SIZE		(WRD_SIZE),
		.WR_WIDTH		(WR_WIDTH),
		.RD_WIDTH		(RD_WIDTH),
		.DEPTH_BIT		(DEPTH_BIT)
	) u_SPI_RXFIFO (
		.i_clk				(i_clk),
		.i_resetn			(i_resetn),

		.i_wreq				(spis_rxfifo_wreq),
		.i_wnwrd			(spis_rxfifo_wnwrd),
		.i_wdata			(spis_rxfifo_wdata),
		.o_wrdy				(spis_rxfifo_wrdy),

		.i_rreq				(spis_rxfifo_rreq),
		.i_rnwrd			(spis_rxfifo_rnwrd[RNWRD_WIDTH-1:0]),
		.o_rrdy				(spis_rxfifo_rrdy),
		.o_rdata			(spis_rxfifo_rdata),
		.o_datacnt			(spis_rxfifo_datacnt)
	);
	assign	spis_rxfifo_full	= (spis_rxfifo_datacnt == 'd16);
	assign	spis_rxfifo_empty	= (spis_rxfifo_datacnt == 'd0);

  // SPI-AHB Interface
	AHB_SFRMAP_BRIDGE # (
		.BUS		(32),
		.BASEADDR	(BASEADDR)
	) u_AHB_SFRMAP_BRIDGE (
		.HCLK			(i_clk),
		.HRESETn		(i_resetn),

		.ahbs_if		(spi_ahbs_if),

		.o_valid_m2s	(valid_m2s),
		.o_haddr		(haddr),
		.o_hsize		(hsize),
		.o_hwrite		(hwrite),
		.o_hwdata		(hwdata),
		.i_hreadyout	(hreadyout),
		.i_hrdata		(hrdata),
		.i_hresp		(hresp)
	);

	SPI_SFRMAP # (
		.BUS		(32)
	) u_SPI_SFRMAP (
		.i_clk			(i_clk),
		.i_resetn		(i_resetn),
	
		.i_valid		(valid_m2s),
		.i_haddr		(haddr),
		.i_hsize		(hsize),
		.i_hwrite		(hwrite),
		.i_hwdata		(hwdata),
		.o_hreadyout	(hreadyout),
		.o_hrdata		(hrdata),
		.o_hresp		(hresp),

		.i_spis_rst_irq_toext	(spis_rst_irq_toext),
		.o_spis_irq_toext		(SPIIRQ),
		.o_spis_irq_code_toext	(spis_irq_code_toext),
		.o_spis_rst_irq_frmext	(spis_rst_irq_frmext),
		.i_spis_irq_code_frmext	(spis_irq_code_frmext),

		.o_spis_txfifo_wreq		(ahbs_spis_txfifo_wreq),
		.o_spis_txfifo_wnwrd	(ahbs_spis_txfifo_wnwrd),
		.o_spis_txfifo_wdata	(ahbs_spis_txfifo_wdata),
		.i_spis_txfifo_full		(spis_txfifo_full),
		.i_spis_txfifo_empty	(spis_txfifo_empty),
		.i_spis_txfifo_datacnt	({{(8-PTR_WIDTH){1'b0}}, spis_txfifo_datacnt}),

		.o_spis_rxfifo_rreq		(spis_rxfifo_rreq),
		.o_spis_rxfifo_rnwrd	(spis_rxfifo_rnwrd),
		.i_spis_rxfifo_rdata	(spis_rxfifo_rdata),
		.i_spis_rxfifo_full		(spis_rxfifo_full),
		.i_spis_rxfifo_empty	(spis_rxfifo_empty),
		.i_spis_rxfifo_datacnt	({{(8-PTR_WIDTH){1'b0}}, spis_rxfifo_datacnt})
	);

  // SPI DMA Interfaces
	// SPI_DMA_M33
	SPI_DMA_M33	u_SPI_DMA_M33 (
		.i_clk				(i_clk),
		.i_resetn			(i_resetn),

		.i_rx_data_valid	(rx_data_valid),
		.i_rx_cmd_valid		(rx_cmd_valid),
		.i_rx_data_byte		(rx_data_byte),
		.o_tx_data_rdy		(tx_data_rdy_m33),
		.o_tx_data_word		(tx_data_word_m33),

		.o_spi_m33_softrstn	(o_spi_m33_softrstn),
		.o_spi_m33sram_config (o_spi_m33sram_config),
		.o_spi_m33cram_ctrl	(o_spi_m33cram_ctrl),
		.o_spi_m33cram_cs	(o_spi_m33cram_cs),
		.o_spi_m33cram_we	(o_spi_m33cram_we),
		.o_spi_m33cram_addr	(o_spi_m33cram_addr),
		.o_spi_m33cram_wdata	(o_spi_m33cram_wdata),
		.i_spi_m33cram_rdata	(i_spi_m33cram_rdata)
	);

// Logic for TX and RX Data
	// SPI_WR_WORD
	always_ff@(posedge i_clk or negedge i_resetn) begin
		if (!i_resetn) begin
			spi_wrwrd_state	<= SPI_WRWRD_IDLE;
			spi_wrwrd_bytecnt	<= 'd0;

			spis_rxfifo_wreq	<= 1'b0;
			spis_rxfifo_wnwrd	<= 'd0;
			spis_rxfifo_wdata	<= 32'd0;
		end
		else begin
			case(spi_wrwrd_state)
				SPI_WRWRD_IDLE: begin
					spis_rxfifo_wreq	<= 1'b0;
					spis_rxfifo_wnwrd	<= 'd0;
					spis_rxfifo_wdata	<= 32'd0;
					if (rx_cmd_valid && rx_data_byte == SPI_WR_WORD) begin
						spi_wrwrd_state	<= SPI_WRWRD_GETWDATA;
					end
				end
				SPI_WRWRD_GETWDATA: begin
					if (rx_data_valid) begin
						spis_rxfifo_wdata	<= {spis_rxfifo_wdata[23:0], rx_data_byte};
						if (spi_wrwrd_bytecnt == 'd3) begin
							spi_wrwrd_bytecnt	<= 3'd0;
							spi_wrwrd_state	<= SPI_WRWRD_WRITE;
						end
						else begin
							spi_wrwrd_bytecnt	<= spi_wrwrd_bytecnt + 3'd1;
						end
					end
				end
				SPI_WRWRD_WRITE: begin
					spi_wrwrd_state	<= SPI_WRWRD_IDLE;
					spis_rxfifo_wreq	<= 1'b1;
					spis_rxfifo_wnwrd	<= 'd1;
				end
			endcase
		end
	end

	// SPI_RD_WORD, SPI_RXFIFO_INFO and SPI_TXFIFO_INFO; SPI_MRAM_BUSY
	always_ff@(posedge i_clk or negedge i_resetn) begin
		if (!i_resetn) begin
			spi_rdwrd_state	<= SPI_RDWRD_IDLE;
			spi_rdwrd_bytecnt	<= 'd0;
			spi_rdwrd_rdata		<= 32'd0;

			tx_data_valid	<= 1'b0;
			tx_data_byte	<= 8'd0;
		end
		else begin
			case(spi_rdwrd_state)
				SPI_RDWRD_IDLE: begin
					tx_data_valid	<= 1'b0;
					tx_data_byte	<= 8'd0;
					if (rx_cmd_valid) begin
						case(rx_data_byte)
							SPI_RD_WORD: begin
								spi_rdwrd_state	<= SPI_RDWRD_GETRDATA;
								spi_rdwrd_rdata	<= spis_txfifo_rdata;
							end
							SPI_RXFIFO_INFO: begin
								spi_rdwrd_state	<= SPI_RDWRD_GETRDATA;
								spi_rdwrd_rdata	<= {
													{(24-PTR_WIDTH){1'b0}}, 
													spis_rxfifo_datacnt, 
													3'b0, spis_rxfifo_empty, 
													3'b0, spis_rxfifo_full
												   };
							end
							SPI_TXFIFO_INFO: begin
								spi_rdwrd_state	<= SPI_RDWRD_GETRDATA;
								spi_rdwrd_rdata	<= {
													{(24-PTR_WIDTH){1'b0}}, 
													spis_txfifo_datacnt, 
													3'b0, spis_txfifo_empty, 
													3'b0, spis_txfifo_full
												   };
							end
							SPI_IRQ_CODE_TOEXT: begin
								spi_rdwrd_state	<= SPI_RDWRD_GETRDATA;
								spi_rdwrd_rdata	<= spis_irq_code_toext;
								//$display("SPI_IRQ_CODE_TOEXT from SPI Master!");
							end
							SPI_MRAM_BUSY: begin
								spi_rdwrd_state <= SPI_RDWRD_GETRDATA;
								spi_rdwrd_rdata <= {31'd0, i_spi_mram_busy};
							end
						endcase
					end
				end
				SPI_RDWRD_GETRDATA: begin
					spi_rdwrd_state	<= SPI_RDWRD_READ;
					tx_data_valid	<= 1'b1;
					tx_data_byte	<= spi_rdwrd_rdata[32-1:24];
					spi_rdwrd_rdata	<= (spi_rdwrd_rdata << 'd8);
				end
				SPI_RDWRD_READ: begin
					if (rx_data_valid) begin
						tx_data_valid	<= 1'b1;
						tx_data_byte	<= spi_rdwrd_rdata[32-1:24];
						spi_rdwrd_rdata	<= (spi_rdwrd_rdata << 'd8);
						if (spi_rdwrd_bytecnt == 3'd2) begin
							spi_rdwrd_bytecnt	<= 3'd0;
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
		spis_txfifo_rreq	= 1'b0;
		spis_txfifo_rnwrd	= 'd0;
		if (rx_cmd_valid && rx_data_byte == SPI_RD_WORD) begin
			spis_txfifo_rreq	= 1'b1;
			spis_txfifo_rnwrd	= 'd1;
		end
	end

// Logic for handling Interrupt
	// SPI_RST_IRQ_TOEXT (Assertion)
	always_ff@(posedge i_clk or negedge i_resetn) begin
		if (!i_resetn) begin
			spis_rst_irq_toext	<= 1'b0;
		end
		else if (rx_cmd_valid) begin
			if (rx_data_byte == SPI_RST_IRQ_TOEXT) begin
				spis_rst_irq_toext	<= 1'b1;
			end
			else begin
				spis_rst_irq_toext	<= 1'b0;
			end
		end
		else begin
			spis_rst_irq_toext	<= 1'b0;
		end
	end

	// SPI_IRQ_FRMEXT (Assertion)
	always_ff@(posedge i_clk or negedge i_resetn) begin
		if (!i_resetn) begin
			spis_irq_frmext	<= 1'b0;
		end
		else begin
			case(spis_irq_frmext)
				1'b0: begin
					if (rx_cmd_valid && (rx_data_byte == SPI_IRQ_FRMEXT)) begin
						spis_irq_frmext	<= 1'b1;
						//$display("SPI IRQ From External Host!");
					end
				end
				1'b1: begin
					if (spis_rst_irq_frmext) begin
						spis_irq_frmext	<= 1'b0;
					end
				end
			endcase
		end
	end
	assign o_spi_m33_irq	= spis_irq_frmext;

	// SPI_IRQ_CODE_FRMEXT (Write data)
	always_ff@(posedge i_clk or negedge i_resetn) begin
		if (!i_resetn) begin
			spis_irq_code_frmext	<= 32'd0;
			spi_icfe_state	<= SPI_ICFE_IDLE;
			spi_icfe_bytecnt	<= 3'd0;
		end
		else begin
			case(spi_icfe_state)
				SPI_ICFE_IDLE: begin
					if (rx_cmd_valid && rx_data_byte == SPI_IRQ_CODE_FRMEXT) begin
						spi_icfe_state	<= SPI_ICFE_GETCODE;
					end
				end
				SPI_ICFE_GETCODE: begin
					if (rx_data_valid) begin
						spis_irq_code_frmext	<= {spis_irq_code_frmext[23:0], rx_data_byte};
						if (spi_icfe_bytecnt == 'd3) begin
							spi_icfe_bytecnt	<= 3'd0;
							spi_icfe_state	<= SPI_ICFE_IDLE;
						end
						else begin
							spi_icfe_bytecnt	<= spi_icfe_bytecnt + 3'd1;
						end
					end
				end
			endcase
		end
	end
endmodule