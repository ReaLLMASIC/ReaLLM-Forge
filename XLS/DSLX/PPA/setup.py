import os
import pdk_info_pb2

import enum
import dataclasses
import json
import pathlib
import subprocess
from typing import Any, Callable, Dict, Optional, Union

# from google.colab import widgets
from google.protobuf import text_format
import pandas as pd

############################################### old ################################################
# yosys = conda_prefix_path / 'bin/yosys'
# openroad = conda_prefix_path / 'bin/openroad'
# yosys_tcl = 'synthesis/synth.tcl'

# default_work_dir = xls.contrib.colab.default_work_dir
####################################################################################################

# conda_prefix_path = os.getenv("CONDA_PREFIX")
conda_prefix_path = pathlib.Path("./conda-env")

# Raise an error if CONDA_PREFIX is not set or the path is invalid
if not conda_prefix_path or not conda_prefix_path.exists():
    raise EnvironmentError("CONDA_PREFIX environment variable is not set or the path is invalid.")

yosys = conda_prefix_path / 'bin' / 'yosys'
openroad = conda_prefix_path / 'bin' / 'openroad'
yosys_tcl = 'synthesis/synth.tcl'

# default_work_dir = pathlib.Path(os.getenv("WORK_DIR"))
default_work_dir = pathlib.Path("./xls_work_dir")
default_work_dir.mkdir(exist_ok=True)

# Raise an error if WORK_DIR is not set or invalid
# if not default_work_dir or not default_work_dir.exists():
#     raise EnvironmentError(f"WORK_DIR environment variable is not set or the path '{default_work_dir}' is invalid.")
####################################################################################################



def pdk_info_proto(
    path: pathlib.Path, optional: bool = False
) -> Optional[pdk_info_pb2.PdkInfoProto]:
    """Load PDK info from prototext.

    Args:
        path: path to prototext file.
        optional: if True, failure to access the pdk info will not produce an error.

    Returns:
        Decoded pdk info proto or None if optional.
    """
    if optional and not path.exists():
        return None
    with path.open('r') as f:
        proto = pdk_info_pb2.PdkInfoProto()
        text_format.Parse(f.read(), proto)
        return proto

pdks = {
    'asap7': {
        'delay_model': 'asap7',
        'pdk_info': pdk_info_proto(
            pathlib.Path('asap7/asap7_data_pdk_info.textproto'),
        ),
    },
}

pdk = 'asap7'

@dataclasses.dataclass(frozen=True)
class RelativeCoreArea:
    utilization_percent: float


@dataclasses.dataclass(frozen=True)
class AbsoluteCoreArea:
    core_width_microns: int
    core_padding_microns: int


@enum.unique
class ImplementationStep(enum.Enum):
    """Steps in the implementation flow."""

    XLS = 'xls'
    SYNTHESIS = 'synthesis'
    PLACEMENT = 'placement'


class PdkRuntimeError(RuntimeError):
    pass


class OpenroadRuntimeError(RuntimeError):
    pass


class OpenstaRuntimeError(RuntimeError):
    pass


class YosysRuntimeError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class SynthesisResults:
    synth_v: pathlib.Path
    design_stats: pd.DataFrame
    cell_stats: pd.DataFrame


def run_synthesis(
    *,
    selected_pdk: Optional[str] = None,
    work_dir: pathlib.Path = default_work_dir,
    silent: bool = False,
) -> SynthesisResults:
    print("hello 124 in setup.py")
    """Run synthesis with Yosys.

    Args:
        selected_pdk: The pdk to use.
        work_dir: Directory that contains verilog and will be where outputs are put.
        silent: Suppress output.

    Returns:
        Metrics from running synthesis.

    Raises:
        PdkRuntimeError: on PDK error.
        YosysRuntimeError: on yosys error.
    """
    if selected_pdk is None:
        selected_pdk = pdk
    pdk_info = pdks[selected_pdk]['pdk_info']
    if pdk_info is None:
        raise PdkRuntimeError(f'PDK "{selected_pdk}" is restricted')

    liberty = (pathlib.Path(pdk) / pathlib.Path(pdk_info.liberty_path).name).resolve()
    synth_v = (work_dir / 'user_module_synth.v').resolve()
    synth_v_flist = (work_dir / 'user_module_synth_v.flist').resolve()
    synth_uhdm_flist = (work_dir / 'user_module_synth_uhdm.flist').resolve()
    synth_uhdm_flist.touch()
    synth_stats_json = (work_dir / 'user_module_synth_stats.json').resolve()
    dont_use_args = ' '.join(
        f'-dont_use {pat}'
        for pat in pdk_info.do_not_use_cell_list
    )

    # run yosys synthesis
    with synth_v_flist.open('w') as f:
        top_v = work_dir / 'user_module.sv'
        f.write(str(top_v.resolve()))
    
    print("hello 161 in setup.py")
    ############################################### old ################################################
    # !FLIST='{synth_v_flist}' ABC_SCRIPT='' CONSTR='' TOP='user_module' OUTPUT='{synth_v}' UHDM_FLIST='{synth_uhdm_flist}' LIBERTY='{liberty}' STATS_JSON='{synth_stats_json}' DONT_USE_ARGS='{dont_use_args}' {yosys} -c '{yosys_tcl}'
    ####################################################################################################

    # Command to run Yosys
    # Note: Must put "FLIST" "ABC_SCRIPT" ... "DONT_USE_ARGS" first. See https://github.com/YosysHQ/yosys/pull/1555.
    # Note: Before running, you must have /xls_work_dir/user_module.sv ready.
    # Note: "FLIST" "ABC_SCRIPT" ... "DONT_USE_ARGS" must be given as environment variables for yosys command.
    
    # [WRONG Version]
    # yosys_command = [
    #     yosys,
    #     "-D", f"FLIST={synth_v_flist}",
    #     "-D", "ABC_SCRIPT=''",
    #     "-D", "CONSTR=''",
    #     "-D", f"TOP=user_module",
    #     "-D", f"OUTPUT={synth_v}",
    #     "-D", f"UHDM_FLIST={synth_uhdm_flist}",
    #     "-D", f"LIBERTY={liberty}",
    #     "-D", f"STATS_JSON={synth_stats_json}",
    #     "-D", f"DONT_USE_ARGS={dont_use_args}",
    #     "-c", yosys_tcl,
    # ]

    yosys_command = [
        yosys,
        "-c", yosys_tcl,
    ]

    env = os.environ.copy()
    env.update({
        "FLIST": synth_v_flist,
        "ABC_SCRIPT": '',
        "CONSTR": '',
        "TOP": "user_module",
        "OUTPUT": synth_v,
        "UHDM_FLIST": synth_uhdm_flist,
        "LIBERTY": liberty,
        "STATS_JSON": synth_stats_json,
        "DONT_USE_ARGS": dont_use_args,
    })
    print(yosys_command)
    print("hello 180 in setup.py")
    try:
        subprocess.run(
            yosys_command,
            check=True,
            capture_output=silent,
            text=True,
            shell=False,
            env=env  # Passing the updated environment
        )
    except subprocess.CalledProcessError as e:
        raise YosysRuntimeError(f"Yosys synthesis failed: {e.stderr}")
    ####################################################################################################
    exit(0)
    print("hello 192 in setup.py")

    with synth_stats_json.open('r') as f:
        synth_stats = json.load(f)
    design_stats = synth_stats['design']
    cells_stats = design_stats.pop('num_cells_by_type')
    design_stats = pd.DataFrame.from_dict(
        design_stats, orient='index', columns=['cells']
    )
    cells_stats = pd.DataFrame.from_dict(
        cells_stats, orient='index', columns=['stats']
    )

    print("hello 204 in setup.py")
    return SynthesisResults(
        synth_v=synth_v, design_stats=design_stats, cell_stats=cells_stats
    )


def run_opensta(
    *,
    selected_pdk: Optional[str] = None,
    work_dir: pathlib.Path = default_work_dir,
    silent: bool = False,
) -> pd.DataFrame:
    """Run OpenSta and collect timing metrics.

    Args:
        selected_pdk: The pdk to use.
        work_dir: Directory that contains verilog.
        silent: Suppress output.

    Returns:
        Dataframe containing timing report.

    Raises:
        OpenstaRuntimeError: on OpenSTA error.
        PdkRuntimeError: on PDK error.
    """
    if selected_pdk is None:
        selected_pdk = pdk
    pdk_info = pdks[selected_pdk]['pdk_info']
    if pdk_info is None:
        raise PdkRuntimeError(f'PDK "{selected_pdk}" is restricted')

    liberty = pathlib.Path(pdk) / pdk_info.liberty_path
    tech_lef = pathlib.Path(pdk) / pdk_info.tech_lef_path
    read_cell_lefs = '\n'.join(
        f'read_lef {pathlib.Path(pdk) / cell_lef_path}'
        for cell_lef_path in pdk_info.cell_lef_paths
    )
    synth_v = work_dir / 'user_module_synth.v'
    top = 'user_module'
    opensta_log = work_dir / 'user_module_sta.log'

    openroad_script = f"""
        sta::redirect_file_begin {opensta_log}
        read_lef {tech_lef}
        {read_cell_lefs}
        read_liberty {liberty}
        read_verilog {synth_v}
        link_design  {top}
        report_checks -unconstrained
        sta::redirect_file_end
    """
    openroad_tcl = work_dir / 'openroad_sta.tcl'
    with openroad_tcl.open('w') as f:
        f.write(openroad_script)

    # run opensta static timing analysis

    ############################################### old ################################################
    # !{openroad} {openroad_tcl} -exit
    ####################################################################################################
    try:
        subprocess.run(
            [openroad, str(openroad_tcl), "-exit"],
            check=True,
            capture_output=silent,
            text=True,
            shell=False,
        )
    except subprocess.CalledProcessError as e:
        raise OpenstaRuntimeError(f"OpenSTA analysis failed: {e.stderr}")
    ####################################################################################################

    columns = ['delay', 'time', 'edge', 'net', 'gate']

    import re
    def sta_report_paths(opensta_log):
        with open(opensta_log) as f:
            sta_report = f.read()
        m = re.search(r'---+(.*)---+', sta_report, flags=re.M | re.S)
        for path in m.group(1).split('\n')[1:-2]:
            parts = path.split(None, maxsplit=len(columns) - 1)
            yield float(parts[0]), float(parts[1]), parts[2], parts[3], parts[4]

    df = pd.DataFrame.from_records(sta_report_paths(opensta_log), columns=columns)
    df['gate'] = df['gate'].str.replace('[()]', '', regex=True)

    return df


@dataclasses.dataclass(frozen=True)
class PlacementResults:
    openroad_global_placement_layout: pathlib.Path
    area: pd.DataFrame
    metrics: pd.DataFrame
    power: pd.DataFrame


def run_placement(
    *,
    clock_period_ps: int,
    placement_density: float,
    core_area: Union[RelativeCoreArea, AbsoluteCoreArea],
    selected_pdk: Optional[str] = None,
    work_dir: pathlib.Path = default_work_dir,
    silent: bool = False,
) -> PlacementResults:
    """Run OpenRoad placement.

    Args:
        clock_period_ps: Clock period in picoseconds.
        placement_density: Placement density in [0.0, 1.0].
        core_area: Relative or absolute core area specification.
        selected_pdk: The pdk to use.
        work_dir: Directory that contains verilog and will be where outputs are put.
        silent: Suppress output.

    Returns:
        Outputs from running placement.

    Raises:
        OpenroadRuntimeError: on OpenRoad error.
        OpenstaRuntimeError: on OpenSTA error.
        PdkRuntimeError: on PDK error.
        ValueError: on invalid inputs.
        YosysRuntimeError: on yosys error.
    """
    clock_period_ns = clock_period_ps / 1000.0
    if selected_pdk is None:
        selected_pdk = pdk
    pdk_info = pdks[selected_pdk]['pdk_info']
    if pdk_info is None:
        raise PdkRuntimeError(f'PDK "{selected_pdk}" is restricted')

    liberty = pathlib.Path(pdk) / pdk_info.liberty_path
    tech_lef = pathlib.Path(pdk) / pdk_info.tech_lef_path
    read_cell_lefs = '\n'.join(
        f'read_lef {pathlib.Path(pdk) / cell_lef_path}'
        for cell_lef_path in pdk_info.cell_lef_paths
    )

    if isinstance(core_area, AbsoluteCoreArea):
        die_side_microns = (
            core_area.core_width_microns + core_area.core_padding_microns * 2
        )
        core_side_microns = (
            core_area.core_width_microns + core_area.core_padding_microns
        )
        initialize_floorplan_args = (
            f' -die_area "0 0 {die_side_microns} {die_side_microns}" -core_area'
            f' "{core_area.core_padding_microns} {core_area.core_padding_microns} {core_side_microns} {core_side_microns}"'
        )
    elif isinstance(core_area, RelativeCoreArea):
        initialize_floorplan_args = (
            f' -utilization {core_area.utilization_percent} -aspect_ratio 1.0'
        )
    else:
        raise ValueError(
            'Expected core_area to be AbsoluteCoreArea or RelativeCoreArea, got'
            f' {core_area!r}'
        )

    initialize_floorplan_command = (
        f'initialize_floorplan -site "{pdk_info.cell_site}"'
        f' {initialize_floorplan_args}'
    )

    def source_pdk_info_tcl(path):
        return f'source {pathlib.Path(pdk) / path}' if path else ''

    source_tracks_file = source_pdk_info_tcl(pdk_info.tracks_file_path)
    source_rc_script_configuration = source_pdk_info_tcl(
        pdk_info.rc_script_configuration_path
    )
    source_pdn_config = source_pdk_info_tcl(pdk_info.pdn_config_path)
    if pdk_info.tapcell_tcl_path:
        tapcell_command = source_pdk_info_tcl(pdk_info.tapcell_tcl_path)
    else:
        tapcell_command = (
            f'tapcell -distance {pdk_info.tapcell_distance} -tapcell_master'
            f' {pdk_info.tap_cell}'
        )

    synth_v = work_dir / 'user_module_synth.v'
    openroad_metrics = work_dir / 'openroad_metrics.json'
    openroad_global_placement_layout = work_dir / 'openroad_global_placement.png'

    openroad_script = f"""
        read_lef {tech_lef}
        {read_cell_lefs}
        read_liberty {liberty}
        read_verilog {synth_v}
        link_design user_module
        {initialize_floorplan_command}
        {source_tracks_file}
        insert_tiecells {pdk_info.tie_high_port} -prefix "TIE_ONE_"
        insert_tiecells {pdk_info.tie_low_port} -prefix "TIE_ZERO_"
        create_clock [get_ports clk] -period {clock_period_ns}
        {source_rc_script_configuration}
        set_wire_rc -signal -layer "{pdk_info.wire_rc_signal_metal_layer}"
        set_wire_rc -clock  -layer "{pdk_info.wire_rc_clock_metal_layer}"
        place_pins -hor_layers {pdk_info.pin_horizontal_metal_layer} -ver_layers {pdk_info.pin_vertical_metal_layer}
        {tapcell_command}
        {source_pdn_config}
        pdngen -verbose
        global_placement -timing_driven -routability_driven -density {placement_density} -pad_left {pdk_info.global_placement_cell_pad} -pad_right {pdk_info.global_placement_cell_pad}
        remove_buffers
        estimate_parasitics -placement
        repair_design
        repair_timing
        utl::metric "utilization_percent" [rsz::utilization]
        utl::metric "design_area" [rsz::design_area]
        utl::metric "power" [sta::design_power [sta::parse_corner {{}}]]
        utl::metric "wns" [sta::worst_slack -max]
        report_power
        report_design_area
        if {{[info procs save_image] == "save_image"}} {{
            save_image -resolution 0.005 "{openroad_global_placement_layout}"
        }}
    """
    openroad_tcl = work_dir / 'place.tcl'
    with openroad_tcl.open('w') as f:
        f.write(openroad_script)
    
    ############################################### old ################################################
    # !QT_QPA_PLATFORM=minimal {openroad} -metrics {openroad_metrics} -exit {openroad_tcl}
    ####################################################################################################
    # Run OpenRoad
    try:
        subprocess.run(
            ["QT_QPA_PLATFORM=minimal", openroad, "-metrics", str(openroad_metrics), "-exit", str(openroad_tcl)],
            check=True,
            capture_output=silent,
            text=True,
            shell=False,
        )
    except subprocess.CalledProcessError as e:
        raise OpenroadRuntimeError(f"OpenRoad placement failed: {e.stderr}")
    ####################################################################################################

    with open(work_dir / 'openroad_metrics.json', 'r') as f:
        metrics = json.loads(f.read())
    df_area = pd.DataFrame.from_dict(
        {
            'global placement': [
                float(metrics['design_area']) * 1e12,
                float(metrics['utilization_percent']) * 100,
            ]
        },
        columns=['area', 'utilization'],
        orient='index',
    )
    metrics_power = [float(m) * 1e6 for m in metrics['power'].split(' ')]
    df_power = pd.DataFrame().from_dict(
        {
            'sequential': metrics_power[4:8],
            'combinational': metrics_power[8:12],
            'clock': metrics_power[12:16],
            'macro': metrics_power[16:20],
            'pad': metrics_power[20:],
            'total': metrics_power[0:4],
        },
        orient='index',
        columns=['internal', 'switching', 'leakage', 'total'],
    )
    df_metrics = (
        pd.DataFrame.from_records([metrics])
        .transpose()
        .set_axis(['metrics'], axis=1)
    )
    return PlacementResults(
        openroad_global_placement_layout=openroad_global_placement_layout,
        area=df_area,
        metrics=df_metrics,
        power=df_power,
    )