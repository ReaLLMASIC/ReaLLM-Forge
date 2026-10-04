DESIGN = fmul
TESTBENCH = ../tb/$(DESIGN)_tb.v
SIM_FILES = ../rtl/define.vh ../rtl/$(DESIGN).sv

VV         = vcs
VVOPTS     = -o $@ +v2k +vc -sverilog -timescale=1ns/1ps +vcs+lic+wait +multisource_int_delays                    \
	       	+neg_tchk +incdir+$(VERIF) +plusarg_save +overlap +warn=noSDFCOM_UHICD,noSDFCOM_IWSBA,noSDFCOM_IANE,noSDFCOM_PONF -full64 -cc gcc +libext+.v+.vlib+.vh
VVOPTS += +define +DUMP_VCD=1 -debug_acc+all -debug_region+cell+encrypt +memcbk +vcs+dumparrays +sdfverbose

all: clean 

sim: 
	$(VV) $(VVOPTS) $(SIM_FILES) $(TESTBENCH); ./$@

clean:
	rm -f ucli.key
	rm -f sim
	rm -f sim_synth
	rm -rf sim.daidir
	rm -rf sim_synth.daidir
	rm -rf csrc
	rm -f *.log
