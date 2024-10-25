"""
Author : Chiara Boscarino
Created on : 2024.06.06
"""

# IMPORTS
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pycodes.modules.stimulus_utils import rescale_sequence
from pycodes.modules import colors


def centers_range(length, bordergap):
    start = bordergap
    stop = (length - bordergap) + 1
    return start, stop


def generate_stimulus(width, height, spot_size, ptg_error, arrangement):
    """ Generate the stimulus for a specific size of the disk, given the arrangement parameters, namely:
        - stimulation area (width x height)
        - disk arrangement
        - max alignment error allowed (expressed as a percentage of the disk size)

        Returns 2 values:
        - matrix of coordinates (N,2)
        - N
    """
    error_diameter = spot_size * ptg_error
    error_radius = error_diameter / 2

    # define the centers distance according to the arrangement
    if arrangement == "grid":
        # disk displacement on a squared grid titling the space
        x = error_radius * 2 / math.sqrt(2); y = x
    elif arrangement == "radial":
        # disk displacement on a equilateral triangular grid titling the space
        x = error_radius * math.sqrt(3); y = error_radius * 3 / 2
    else:
        print("Error: arrangement not recognized"); return -1

    coordinates = []
    i = 0
    shift = True
    small_spots_upbound = 250
    big_spots_lowbound = 400
    very_big_spots_lowbound = 1000
    # relax the error alignment constraint at the borders for small spots to reduce the total number of spots
    # and so keep the overall duration feasible
    if spot_size < small_spots_upbound:
        starty, endy = centers_range(height, 40)
        startx,endx = centers_range(width, 40)

    # tighten the error alignment constraint for bigger spots, to avoid loosing big areas of cells,
    # since this won't affect significantly the overall duration
    elif big_spots_lowbound < spot_size < very_big_spots_lowbound:
        starty, endy = centers_range(height, 0.25*error_radius)
        startx, endx = centers_range(width, 0.25*error_radius)

    elif spot_size > very_big_spots_lowbound:
        starty = 0 ; endy = height+y
        startx= 0 ; endx = width+x

    else:
        starty, endy = centers_range(height, error_radius)
        startx, endx = centers_range(width, error_radius)

    for Y in np.arange(starty, endy, y):
        rowstartx = startx + x / 2 if shift else startx
        for X in np.arange(rowstartx, endx, x):
            coordinates.append([X,Y])
            i += 1
        shift = not shift

    return np.array(coordinates), i


def simulate_stimulus(width, height, spot_sizes, ptg_error, arrangement):
    """ Given a stimulation area (width x height), a specific RANGE of disk sizes, a disk arrangement on the area
        and a max alignment error allowed (expressed as a percentage of the disk size), simulate the disk displacement,
        and return the computed set of coordinates and the total number of spots for ALL disk sizes.
        Returns 2 dictionaries:
        - coords[disk size] = matrix of coordinates (N,2) --> i-th row in the matrix contain the (x,y)
                                                            coordinates of the center of the i-th disk
        - tot_spots[disk size] = N
    """
    coords = {}
    tot_spots = {}
    for i, spot_size in enumerate(spot_sizes):
        coordinates, tot_spot = generate_stimulus(width, height, spot_size, ptg_error, arrangement)
        coords[spot_size] = coordinates
        tot_spots[spot_size] = tot_spot

    return coords, tot_spots


def one_size_duration(k_spot_repetitions, tot_num_spot, spot_onset_time, spot_offset_time):
    # This computes the duration of the stimulation of a single spot size
    # meaning the time necessary to show each spot on the area for k_spot_repetitions times
    # considering that each spot is shown for spot_onset_time and followed by a spot_offset_time of darkness

    return (spot_onset_time + spot_offset_time) * k_spot_repetitions * tot_num_spot


def estimate_duration(tot_spots, k, estimation_id, spot_onset_time, spot_offset_time, output=None):
    """ Given the total number of spots for each size, the number of repetitions for each spot,
        the onset and offset duration for each spot, estimate the duration of the stimulus
        for each size and in total.

        Args:
        - tot_spots: dictionary containing the total number of spots for each size
        - k: number of repetitions for each spot
        - estimation_id: identifier for the output
        - spot_onset_time: duration of the spot onset
        - spot_offset_time: duration of the spot offset
        - output: in ["print", "csv", None]

        Returns the total duration in seconds, minutes and hours.
    """
    df = {"Spot size": [],
          "Tot spot": [],
          "Duration": [] }

    for size, n in tot_spots.items():
        tot_dur = one_size_duration(k, n, spot_onset_time, spot_offset_time)
        df["Spot size"].append(size)
        df["Tot spot"].append(n)
        df["Duration"].append(datetime.timedelta(seconds=tot_dur))

    df = pd.DataFrame.from_dict(df)
    if output == "print":
        print(f"\nESTIMATION FOR {estimation_id}")
        print(df)
    elif output == "csv":
        df.to_csv(estimation_id, index=False)
    elif output is None:
        pass
    else:
        raise ValueError("output must be 'print', 'csv' or None")

    tot_dur = np.sum(df["Duration"])

    tot_dur_s = tot_dur.total_seconds()
    tot_dur_m = tot_dur_s / 60
    tot_dur_h = tot_dur_m / 60

    return tot_dur_s, tot_dur_m, tot_dur_h


def plot_stimulus(width, height, ptg_error, coordinates, fnimg=None,
                  visualize_stimulation_area=False, target_area_shift=None, total_area_width=None, total_area_height=None, mea_size=None):
    """ Given a set of coordinates and the parameters of a stimulation area, plot the expected stimulus
    """
    if visualize_stimulation_area:
        if target_area_shift is None or total_area_width is None or total_area_height is None or mea_size is None:
            raise ValueError("Provide the required parameters for the visualization of the stimulation area:\n"
                             "target_area_shift, total_area_width, total_area_height, mea_size")
        # adjust plot dimensions
        w = total_area_width / 300 ; h = total_area_height / 300
        total_area_center = (total_area_width / 2, total_area_height / 2)
        mea_top_left_corner = (total_area_center[0] - mea_size / 2, total_area_center[1] - mea_size / 2)
    else:
        # adjust plot dimensions
        w = width / 100 ; h = height / 100

    ss = coordinates.keys()
    num = len(ss)
    fg, axs = plt.subplots(1, num, figsize=(num * (w + 1), h))
    axs = axs.flatten()

    for i, size in enumerate(ss):
        error_radius = size * ptg_error / 2
        coords = np.array(coordinates[size])
        n = coords.shape[0]
        ax = axs[i]

        # set full circles to show
        full_spot_to_show = [0, 1, 2, 3, n-4, n-3, n-2, n-1]
        for j, (x, y) in enumerate(coords):

            if visualize_stimulation_area:
                (x, y) = (x + target_area_shift, y + target_area_shift)
                ax.set_xlim(0, total_area_width)
                ax.set_ylim(total_area_height, 0)
                ax.set_aspect("equal")
                # add mea border
                ax.add_artist(plt.Rectangle(mea_top_left_corner, mea_size, mea_size, ec="gray", lw=2, fc="none"))

            else:
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal")

            # Add spot margin error, full spot (if in the list) and spot center
            ax.add_artist(plt.Circle((x, y), error_radius, ec="steelblue", fc="lightskyblue", alpha=0.4))
            if j in full_spot_to_show:
                ax.add_artist(plt.Circle((x, y), size / 2, ec="#0838a1", fc="#0838a1", alpha=0.1))
            ax.add_artist(plt.Circle((x, y), 1, fc="black"))

        ax.set_title(f"Spot size: {size} (Tot spots: {n})")

    plt.show()
    if fnimg is not None:
        plt.savefig(fnimg)
    return


def plot_alignment_cases(rf_diameter, spot_size, ptg_error_margin):
    """ Plot best-worst case scenario for the cell-spot alignment for the spot stimulus.
    """
    rf_radius = rf_diameter / 2
    spot_radius = spot_size / 2

    # Define the error radius
    error_radius = spot_size * ptg_error_margin / 2

    # Define the scenarios
    scenarios = {"Best": 0, "Medium": 0.5, "Worst": 1}

    plot_area = 3 * rf_radius
    rf_center = (plot_area / 2, plot_area / 2)
    rf_color = colors.BLUE
    spot_color = colors.ORANGE
    fontsize = 12

    n_rows = 1
    n_cols = len(scenarios)
    sq_dim = 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * sq_dim, n_rows * sq_dim))
    axs = axs.flatten()
    fig.suptitle(f"Cell-Spot Alignment Scenarios with ptg error margin {ptg_error_margin}\nRF diameter {rf_diameter} um, Spot size {spot_size} um", fontsize=fontsize)

    for i, (scenario, ptg) in enumerate(scenarios.items()):
        ax = axs[i]
        ax.set_xlim(0, plot_area)
        ax.set_ylim(plot_area, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{scenario} (d={ptg}*max_d)", fontsize=fontsize)

        cell_spot_distance = error_radius*ptg
        spot_center = (rf_center[0] + cell_spot_distance, rf_center[1])

        # Plot the RF with center
        ax.add_artist(plt.Circle(rf_center, rf_radius, ec=rf_color, lw=2, fc="none"))
        ax.add_artist(plt.Circle(rf_center, 1, ec=rf_color, lw=2, fc="none"))

        # Plot the spot with center
        ax.add_artist(plt.Circle(spot_center, spot_radius, ec="none", lw=2, fc=spot_color))
        ax.add_artist(plt.Circle(spot_center, 1, ec="black", lw=2, fc="none"))

    plt.show()
    return


def create_disk(frame_dimensions,
                center_coordinates,
                diameter,
                background_value=0,
                disk_value=1):
    """
    Create a disk frame with the specified parameters.

    Args:
    - frame_dimensions (tuple): dimensions of the frame (width, height)
    - center_coordinates (tuple): coordinates of the center of the disk (x, y)
    - diameter (int): diameter of the disk
    - background_value (int): value of the background color
    - disk_value (int): value of the disk color

    """

    radius = diameter/2

    ellipse_field = np.zeros((frame_dimensions[1], frame_dimensions[0]))
    for i in range(frame_dimensions[0]):
        for j in range(frame_dimensions[1]):
            ellipse_field[i, j] = (i - center_coordinates[1]) ** 2 / radius ** 2 + (
                        j - center_coordinates[0]) ** 2 / radius ** 2

    disk_mask = (ellipse_field < 1)
    background_mask = (ellipse_field >= 1)

    frame = background_value * np.ones(frame_dimensions) * background_mask + disk_value * np.ones(frame_dimensions) * disk_mask
    return frame


def transform_info(size, x, y, origin):
    actual_size = size
    actual_x, actual_y = np.array([x, y]) + origin
    return actual_size, actual_x, actual_y


def get_frame_info(frames_reference, frame_id):
    """ Retrieve the information about a frame from the frames_reference file.
        The frames_reference file contains the information about the frames of the checkerboard sequence.
        Each row of the file contains the following information about a frame:
        - size (in checks)
        - x coordinate of the center of the frame (in checks)
        - y coordinate of the center of the frame (in checks)
        - x coordinate of the center of the frame (in unit)
        - y coordinate of the center of the frame (in unit)
        - x coordinate of the center of the frame (in unit, aligned to the center of the area)
        - y coordinate of the center of the frame (in unit, aligned to the center of the area)

        Args:
        - frames_reference (np.array): 2D matrix (num_frames, 7) containing the information about the frames
            Note: use the following codes to load the frames_reference file:
                reference_path = "stimulus_radial_error_0.2_frames_reference.vec"
                frames_reference = np.genfromtxt(reference_path)
        - frame_id (int): id of the frame

        Returns:
        - size (int): size of the frame (in checks)
        - x_original (int): x coordinate of the center of the frame (in checks)
        - y_original (int): y coordinate of the center of the frame (in checks)
        - x_actual (float): x coordinate of the center of the frame (in unit)
        - y_actual (float): y coordinate of the center of the frame (in unit)
        - x_aligned (float): x coordinate of the center of the frame (in unit, aligned to the center of the area)
        - y_aligned (float): y coordinate of the center of the frame (in unit, aligned to the center of the area)
    """
    size = frames_reference[frame_id, 0]
    x_original = frames_reference[frame_id, 1]
    y_original = frames_reference[frame_id, 2]
    x_actual = frames_reference[frame_id, 3]
    y_actual = frames_reference[frame_id, 4]
    x_aligned = frames_reference[frame_id, 5]
    y_aligned = frames_reference[frame_id, 6]
    return size, x_original, y_original, x_actual, y_actual, x_aligned, y_aligned


def process_sequence(original_stack, rescale_factor_, margin_):
    """Process the stack of spot frames to align with the MSF processing.
       Args:
        - stack: stack of spot frames
        - rescale_factor: rescale factor to be applied to the sequence to max reduce redundancy
        - margin: margin to be cropped from the sequence (all around)
    """
    # Rescale to have the min number of pixels per check avoiding redundancy
    sequence_rescaled = rescale_sequence(original_stack, n_pixel_per_check_old=rescale_factor_, n_pixel_per_check_new=1)

    # Keep only the portion corresponding to the stimulated area with the spots
    sequence_cropped = sequence_rescaled[:, margin_:-margin_, margin_:-margin_]

    processed_stack = np.array(sequence_cropped)
    return processed_stack
