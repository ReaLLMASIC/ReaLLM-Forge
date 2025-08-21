import setup
import pandas as pd
from IPython.display import display
import PIL.Image

def main():
    # User inputs and configurations
    placement_density = 0.8  # Replace with user input or sliders
    clock_period_ps = 800  # Replace with user input or sliders
    clock_period_ns = clock_period_ps / 1000.0

    core_area = 'relative'  # Change to "absolute" for absolute area configuration
    utilization_percent = 50  # Utilization percentage for relative area
    core_width_microns = 600  # Width for absolute area
    core_padding_microns = 60  # Padding for absolute area

    # Determine core area configuration
    if core_area == 'relative':
        core_area_value = setup.RelativeCoreArea(utilization_percent)
    else:
        core_area_value = setup.AbsoluteCoreArea(core_width_microns, core_padding_microns)

    # Run synthesis
    print("Running synthesis...")
    print("hello 25 ppb.py")
    synth_results = setup.run_synthesis()
    print("hello 27 ppb.py")
    display_synthesis_results(synth_results)

    # Run static timing analysis
    print("Running static timing analysis...")
    timing_results = setup.run_opensta()
    display_timing_results(timing_results)

    # Run placement
    print("Running placement...")
    placement_results = setup.run_placement(
        clock_period_ps=clock_period_ps,
        placement_density=placement_density,
        core_area=core_area_value
    )
    display_placement_results(placement_results)

def display_synthesis_results(synth_results):
    """Display synthesis results."""
    print("hello 44 ppb.py")
    print("Synthesis Results:")
    print("Cell Stats:")
    display(synth_results.cell_stats)
    print("Design Stats:")
    display(synth_results.design_stats)
    print("Gate-Level Netlist:")
    with synth_results.synth_v.open('r') as f:
        print(f.read())

def display_timing_results(timing_results):
    """Display static timing analysis results."""
    print("Static Timing Analysis Results:")
    styled_timing = (
        timing_results.style
        .hide(axis='index')
        .background_gradient(subset=['delay'], cmap='Oranges')
        .bar(subset=['time'], color='lightblue')
    )
    display(styled_timing)

def display_placement_results(placement_results):
    """Display placement results including area, power, and layout."""
    print("Placement Results:")
    
    # Display area
    print("Area Estimate:")
    styled_area = placement_results.area.style.format("{:.3f} μm²", subset=['area'])
    styled_area = styled_area.format("{:.2f} %", subset=['utilization'])
    styled_area = styled_area.bar(subset=['utilization'], color='lightblue', vmin=0, vmax=100)
    display(styled_area)

    # Display power metrics
    print("Power Metrics:")
    styled_power = (
        placement_results.power.style
        .format("{:.3f} uW")
        .background_gradient(cmap='Oranges', axis=None)
        .bar(subset=['total'], color='lightcoral')
    )
    display(styled_power)

    # Display global placement layout
    print("Global Placement Layout:")
    if placement_results.openroad_global_placement_layout.exists():
        img = PIL.Image.open(placement_results.openroad_global_placement_layout)
        img = img.resize((500, 500))
        display(img)

if __name__ == "__main__":
    main()