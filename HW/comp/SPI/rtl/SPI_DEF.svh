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

/////////////////////////////////////////////////////////////////////////
//                                                                     //
//   Filename :  SPI_DEF.svh                                           //
//                                                                     //
//  Description :  This file has the macro-defines for macros used in  //
//                 the SPI Slave design.                               //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

`ifndef __SPI_DEFS_SVH__
`define __SPI_DEFS_SVH__

//////////////////////////////////////////////
//
// Attribute definitions
//
//////////////////////////////////////////////

// SPI Command Set. Give a 16-CMD space for each block now. 
typedef enum	logic	[8-1:0]	{
	SPI_NOP				= 8'd0,

	// Consmax_lut
	SPI_CONS_WR_WORD    = 8'h01, // 8b-CMD + 5b-addr + 16b-wdata 1
	
	// TODO: Inst_reg 
	SPI_INST_WR_WORD	= 8'h02, // 8b-CMD + 2b-addr + 3b-wdata 1
	SPI_INST_RD_WORD 	= 8'h03, // 8b-CMD + 2b-addr

	// SPI SRAM
	SPI_SRAM_WR_WORD	= 8'h04, // 8b-CMD + 13b-addr + 64b-wdata 1
	SPI_SRAM_RD_WORD	= 8'h05, // 8b-CMD + 13b-addr
	//MSB 0 for SRAM0, MSB 1 for SRAM1

	// SPI CFG
	SPI_CFG_WR_WORD     = 8'h06, // 8b-CMD + 2b-addr  + 56b-data

	// START
	SPI_START_WR_WORD   = 8'h07, // 8b-CMD

	// GBUS
	SPI_GBUS_WR_WORD	= 8'h08, // 8b-CMD + 18b-addr + 64b-data
	SPI_GBUS_RD_WORD	= 8'h09,  // 8b-CMD + 18b-addr

	// CLK_CFG
	SPI_PLL_CFG_WR_WORD = 8'h0a //8b-CMD + 128b-cfg
}	SPI_CMD;

// TX and RX FIFO Access FSMs
// SPI_WR_WORD
typedef	enum	logic	[4-1:0]	{
	SPI_WRWRD_IDLE		   =    4'd0,
	SPI_WRWRD_GET_CONS	   =    4'd1,
	SPI_WRWRD_GET_INST     =    4'd2,
	SPI_WRWRD_GET_SRAM     =    4'd3,
	SPI_WRWRD_GET_CFG      =    4'd4,
	SPI_WRWRD_GET_START    =    4'd5,
	SPI_WRWRD_GET_GBUS     =    4'd6,
	SPI_WRWRD_GET_PLL_CFG  =    4'd7,
	SPI_WRWRD_WRITE		   =    4'd8
}	SPI_WRWRD_FSM;

// SPI_RD_WORD, SPI_RXFIFO_INFO and SPI_TXFIFO_INFO
typedef enum	logic	[4-1:0]	{
	SPI_RDWRD_IDLE		= 4'd0,
	SPI_RDWRD_GET_INST	= 4'd1,
	SPI_GET_INST_ADDR   = 4'd2,
	SPI_RDWRD_GET_SRAM  = 4'd3,
	SPI_GET_SRAM_ADDR   = 4'd4,
	SPI_RDWRD_GET_GBUS  = 4'd5,
	SPI_GET_GBUS_ADDR   = 4'd6,
	SPI_RDWRD_SRAM		= 4'd7,
	SPI_RDWRD_GBUS      = 4'd8
}	SPI_RDWRD_FSM;

`endif
