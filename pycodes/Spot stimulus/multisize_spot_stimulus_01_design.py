"""
Author : Chiara Boscarino
Created on : 2024.06.06


DESIGN of TILING MULTI-SIZE SPOT STIMULUS

This script enables the design of the Multi-size spot stimulus for MEA retinal recordings.
The Multi-size spot stimulus is a stimulus that enables the stimulation with multi-size spots
of a portion of the retina using MEA recordings.

The stimulus is designed to show to the target piece of retina the spots required to stimulate
each cell with spots of all defined SIZES, each ALIGNED with the center the cell and repeated K times
to be able to compute an average response (reduce noise and estimate cell's repeatability).

Each spot is identified by its SIZE (spot diameter) and LOCATION.
Spots are presented one at a time for a determined ON DURATION,
and followed by a blank frame lasting a determined OFF DURATION.
The overall stimulus is a sequence of ON SPOT and OFF frames, each related to a specific
spot (of size s) stimulating at a specific location (spot_center_x, spot_center_y) the present RGCs.

Not knowing a priori the location of the cells on the target portion of retina,
to be sure that each recorded cell has been stimulated with K spots of each size
ALIGNED with its soma, we defined an optimized tiling arrangement.
It minimizes the number of spots required to be sure that at least one spot of each size
must be centered within an ADMITTED MARGIN of ALIGNMENT around the cell body, wherever it is.
Thus, it tiles the target area with spots so any location in the target area falls within the
radial ADMITTED MARGIN of ALIGNMENT.
Notes:
(1) the ADMITTED ALIGNMENT MARGIN is defined as a percentage of the spot size,
refer to the repo resources for further explanations.
(2) here are envisaged two possible arrangements, squared or radial grid for spot centers location.
While we have proven the higher efficacy of the radial option, we provide estimation for both for completeness.

This script enables the definition of the ARRANGEMENT and NUMBER of REPETITIONS (K) for each spot
as a trade-off between achievable SNR after average and DURATION of the stimulus,
given the total required number of spots required for tiling.

USER GUIDE:
1. Define the parameters;
2. Run the codes to estimate the duration of the stimulus for the different options;
3. Analyse the results and define the optimal set of parameters
NEXT --> go to multisize_spot_stimulus_02_1_generation.py to generate the stimulus

"""

# 1. PARAMETERS
# Stimulus parameters
SIZES = [150, 250, 600, 1200]  # diameter (microns)
TARGET_AREA_WIDTH = 1500  # microns
TARGET_AREA_HEIGHT = 1500  # microns

ON_DURATION = 1  # (seconds)
OFF_DURATION = 1  # (seconds)
INITIAL_DELAY = 0  # (seconds)

MAX_ADMITTED_ALIGNMENT_MARGIN_PTGs = [0.45]  # *100% of spot size (or [0.1, 0.15, 0.2, 0.25,  0.3])
Ks = range(2, 15, 2)  # number of repetitions of each spot size (or range(1,15))

# Output analysis parameters
time_mod = "minutes"  # in ["seconds", "minutes", "hours"]
arrangements = ["radial"]  # in ["grid", "radial"]
max_duration_m = 240  # (minutes)
MARKERSIZE = 7
LINEWIDTH = 2
ALPHA = 0.08
LABELSIZE = 18
TICKSSIZE = 16
TITLESIZE = 22
LEGENDSIZE = 20
LABELPAD = 20
TITLEPAD = 30


# Reports to show
SHOW_DURATION_ESTIMATION = True
SHOW_SINGLE_ARRANGEMENT = True

# if SHOW_SINGLE_ARRANGEMENT select the MAX_ADMITTED_ALIGNMENT_MARGIN_PTG
max_admitted_alignment_margin_ptg = 0.45
arrangement = "radial"
alignment_case_rf_diameter = 120  # microns
alignment_case_spot_size = SIZES[0]  # microns
SPOT_STIMULUS_DIMENSION_px = 768
PIXEL_SIZE = 3.5
N_ELECTRODES = 16
ELECTRODE_SPACING = 100  # microns
MEA_SIZE_um = 1530  # N_ELECTRODES * ELECTRODE_SPACING
SPOT_STIMULUS_DIMENSION_um = SPOT_STIMULUS_DIMENSION_px * PIXEL_SIZE
SPOT_STIMULUS_CENTER_COORDINATE_um = SPOT_STIMULUS_DIMENSION_um / 2  # center of the stimulus (starting from the top left corner of the stimulus)
SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um = SPOT_STIMULUS_CENTER_COORDINATE_um - MEA_SIZE_um / 2  #

# 2. ANALYSIS
from pycodes.modules.multisize_spot_stimulus_utils import *
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('axes', labelsize=LABELSIZE)
plt.rc('axes', titlesize=TITLESIZE)
plt.rc('xtick', labelsize=TICKSSIZE)
plt.rc('ytick', labelsize=TICKSSIZE)
plt.rc('legend', fontsize=LEGENDSIZE)
max_duration_s = max_duration_m * 60  # (seconds)

if SHOW_SINGLE_ARRANGEMENT:
    # Generate coordinates
    coords, tot_spots = simulate_stimulus(TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT, SIZES,
                                          max_admitted_alignment_margin_ptg, arrangement)
    # Show the stimulus arrangement
    plot_stimulus(TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT, max_admitted_alignment_margin_ptg, coords,
                  visualize_stimulation_area=True, target_area_shift=SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um,
                  total_area_width=SPOT_STIMULUS_DIMENSION_um, total_area_height=SPOT_STIMULUS_DIMENSION_um, mea_size=MEA_SIZE_um)
    # # Plot alignment cases
    plot_alignment_cases(alignment_case_rf_diameter, alignment_case_spot_size, max_admitted_alignment_margin_ptg)

if SHOW_DURATION_ESTIMATION:
    # Compute duration for each option
    estimations = {}
    for arrangement in arrangements:
        estimations[arrangement] = {}
        for max_error in MAX_ADMITTED_ALIGNMENT_MARGIN_PTGs:
            estimations[arrangement][max_error] = []
            for k in Ks:
                _, tot_spots = simulate_stimulus(TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT, SIZES, max_error, arrangement)
                tot_dur_s, tot_dur_m, tot_dur_h = estimate_duration(tot_spots, k, f"estimation_{arrangement}_{k}rep.csv", ON_DURATION, OFF_DURATION, output=None)
                if time_mod == "seconds":
                    estimations[arrangement][max_error].append(tot_dur_s)
                elif time_mod == "minutes":
                    estimations[arrangement][max_error].append(tot_dur_m)
                elif time_mod == "hours":
                    estimations[arrangement][max_error].append(tot_dur_h)
                else:
                    raise ValueError("time_mod must be 'seconds', 'minutes' or 'hours'")

    # Report the estimation
    df = pd.DataFrame(columns=Ks)
    for arrangement in arrangements:
        for max_error in MAX_ADMITTED_ALIGNMENT_MARGIN_PTGs:
            df.loc[f"{arrangement} arrg; {max_error} marg"] = np.round(estimations[arrangement][max_error], 2)
    print(df)

    # Visualize the estimation
    fig = plt.figure(figsize=(15, 7))
    ax = plt.subplot(111)

    if time_mod == "seconds":
        step = 60*60
        max_dur = max_duration_s
    elif time_mod == "minutes":
        step = 60
        max_dur = max_duration_m
    elif time_mod == "hours":
        step = 3
        max_dur = max_duration_m/60
    else:
        raise ValueError("time_mod must be 'seconds', 'minutes' or 'hours'")

    for arrangement in arrangements:
        for max_error in MAX_ADMITTED_ALIGNMENT_MARGIN_PTGs:
            x = np.array(estimations[arrangement][max_error])
            ax.plot(Ks, x, "-o", markersize=MARKERSIZE, linewidth=LINEWIDTH, label=f"{arrangement} arrg; {max_error} marg")
            xx = x[np.where(x < max_dur)[0]]
            if len(xx) > 0:
                max_index, max_value = max(enumerate(xx), key=lambda y: y[1])
                ax.scatter(Ks[max_index], max_value, s=MARKERSIZE*30, facecolor='none', edgecolor='black', linewidth=LINEWIDTH*0.8)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.axhspan(YLIM[0], max_dur, color="green", alpha=ALPHA)
    # ax.axhspan(max_dur, YLIM[-1], color="red", alpha=ALPHA)
    max_y = ax.get_ylim()[1]
    ax.set_xlim([0, Ks[-1]+1])
    ax.set_ylim([0, max_y])
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)

    ytks = np.arange(step, max_y + step, step).astype(int)
    ax.set_yticks(ytks)
    ax.set_yticklabels(ytks)
    ax.set_ylabel(f"Duration ({time_mod})", fontsize=LABELSIZE, labelpad=LABELPAD)
    ax.set_xlabel("Number of repetitions", fontsize=LABELSIZE, labelpad=LABELPAD)

    ax.set_title(f"Multi-size Spot Stimulus Total Duration ({TARGET_AREA_WIDTH}x{TARGET_AREA_HEIGHT})", fontsize=TITLESIZE, pad=TITLEPAD)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0, frameon=False)
    fig.tight_layout()
    fig.show()
