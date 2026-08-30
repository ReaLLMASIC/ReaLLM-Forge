TESTBENCH = top.sv
SIM_FILES = test.v

VV         = vcs
VVOPTS     = -o $@ +v2k +vc -sverilog -timescale=1ns/1ps +vcs+lic+wait +multisource_int_delays                    \
	       	+neg_tchk +incdir+$(VERIF) +plusarg_save +overlap +warn=noSDFCOM_UHICD,noSDFCOM_IWSBA,noSDFCOM_IANE,noSDFCOM_PONF -full64 -cc gcc +libext+.v+.vlib+.vh
VVOPTS += +define +DUMP_VCD=1 -debug_acc+all -debug_region+cell+encrypt +memcbk +vcs+dumparrays +sdfverbose

all: clean 

sim: 
	cd verilog; $(VV) $(VVOPTS) $(SIM_FILES) $(TESTBENCH); ./$@

clean:
	rm -f verilog/ucli.key
	rm -f verilog/sim
	rm -f verilog/sim_synth
	rm -rf verilog/sim.daidir
	rm -rf verilog/sim_synth.daidir
	rm -rf verilog/csrc
	rm -f verilog/*.log
