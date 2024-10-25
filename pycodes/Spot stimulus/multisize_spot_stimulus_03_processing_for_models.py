"""
Author : Chiara Boscarino
Created on : 2024.06.07


PROCESSING of TILING MULTI-SIZE SPOT STIMULUS to be used as INPUT for MODELS
This script enables the generation of the Multi-size spot stimulus to be used as input for MODEL SPOT PREDICTION.

It includes:
    1. Process the frame stack to make it lighter (rescale to reduce the spatial resolution (nb of pixels per frame)
        + crop to keep only the center of the target area)
    2. Visualize and check the processed stack
    3. Generate the full stimulus video to be used as input for the models


USER GUIDE
1. Choose what you want to compute;
2. Set the parameters;
2. Run the codes to generate the stimulus;

"""

# 1. CHOOSE CODE TO RUN
PROCESS_SPOT_STACK_FOR_MODELS = True
CHECK_SPOT_STACK_FOR_MODELS = True
GENERATE_FULL_SPOT_STIMULUS_FOR_MODELS = True

# 2. SET PARAMETERS
# -- RESCALE SIZE
rescale_factor = 8

# -- CROP SIZE
# Given that in the spot stimulus has been designed to tile only a central target area,
# define the dimension of the margin to be cropped alla around from the original frames to keep only the tiled central area.
# dim_factor is the portion of the margin left out in the original stimulus.
# original margin in the target area (top-left) = SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE
# margin to crop = SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE * dim_factor
dim_factor = 4 / 5

# -- FULL STIMULUS GENERATION FOR MODELS
prediction_frequency = 40  # Hz

# -- FILENAMES
original_spot_stack = "stimuli/multisize_spot_tiling_stimulus_V3/multisize_spot_tiling_stimulus_radial_error_20_V3_frame_stack.npy"
vec_filepath = "stimuli/multisize_spot_tiling_stimulus_V3/multisize_spot_tiling_stimulus_radial_error_20_V3.vec"
processed_spot_stack = original_spot_stack.replace("frame_stack.npy", "frame_stack_processed_for_models.npy")
full_spot_stimulus_for_models = original_spot_stack.replace("frame_stack.npy", f"reconstructed_for_models_{prediction_frequency}Hz.npy")
full_spot_stimulus_for_models_frame_sequence = original_spot_stack.replace("frame_stack.npy", f"reconstructed_for_models_{prediction_frequency}Hz_frame_sequence.npy")

# -- SPOT STIMULUS PARAMETERS
black_screen_frame_id = 0


# 3. CODE
import numpy as np
import os
from pycodes.modules.multisize_spot_stimulus_utils import process_sequence
from pycodes.modules.visual_utils import plot_frame_sequence

from multisize_spot_stimulus_02_0_generation_parameters import *

os.chdir(root)
print(f"Working directory: {os.getcwd()}")

SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_px = int(np.floor(SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um / PIXEL_SIZE))
margin = int((SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_px * dim_factor) / rescale_factor)
expected_new_size = (SPOT_STIMULUS_DIMENSION_px / rescale_factor) - 2 * margin


if PROCESS_SPOT_STACK_FOR_MODELS:

    # Load the spot stimulus
    disk_stimulus = np.load(original_spot_stack)

    # Process the spot stimulus
    processed_stack = process_sequence(disk_stimulus, rescale_factor, margin)

    # Save the processed spot stimulus for the models
    np.save(processed_spot_stack, processed_stack)

    print(f"Processed spot stimulus: \n"
          f"\tOriginal shape: {disk_stimulus.shape}\n"
          f"\tProcessed shape: {processed_stack.shape}\n"
          f"\tSaved at {processed_spot_stack}")


if CHECK_SPOT_STACK_FOR_MODELS:
    print("\n\nChecking the npy stack")
    to_plot = [0, 1, 709, 711, 1022, 1024, 1209, 1211, 1239, 1241, 1253, 1254]
    # Load the processed spot stimulus
    processed_stack = np.load(processed_spot_stack)

    print(f"Processed spot stimulus: \n"
          f"\tShape: {processed_stack.shape}\n"
          f"\tLoaded from {processed_spot_stack}")

    plot_frame_sequence(processed_stack[to_plot], cmap="gray", crange=None)


# if CHECK_SPOT_VIDEO_FOR_MODELS:
#     %matplotlib qt
#     print("\n\nChecking the full video")
#     to_visualize = [1, 709, 711, 1022, 1024, 1209, 1211, 1239, 1241, 1253, 1254]
#     stimulus_video = np.load(full_spot_stimulus_for_models)
#
#     anim, fig = plot_animation(video1=stimulus_video, cmap1="gray", title1="Spot")


if GENERATE_FULL_SPOT_STIMULUS_FOR_MODELS:

    # -------------------- SEQUENCES --------------------- #
    # Get the stimulus sequences from the .vec file
    vec_sequences = np.genfromtxt(vec_filepath)

    # parameters
    sequence_column = 1
    adaptation_length = 0  # frames

    # remove the header and the adaptation sequences and get only the sequence column
    sequence = [int(x) for x in vec_sequences[(1 + adaptation_length):, sequence_column]]
    total_frames = len(sequence)

    # get the shuffled sequence of frames
    frames_sequence = [int(sequence[i]) for i in range(0, len(sequence), SPOT_SEQUENCE_DURATION_frames)]
    frames_sequence_no_rep = []
    for x in frames_sequence:
        if x not in frames_sequence_no_rep:
            frames_sequence_no_rep.append(x)

    # check the number of repetitions
    non_rep_frames = [x for x in sequence if x != black_screen_frame_id]
    frame_ids, count = np.unique(non_rep_frames, return_counts=True)
    nb_repetitions = count / STIMULUS_FREQUENCY
    n_rep, r_count = np.unique(nb_repetitions, return_counts=True)
    if len(n_rep) > 1:
        print(f"WARNING: The number of repetitions is not constant: {dict(zip(frame_ids, nb_repetitions))}")
    else:
        n_rep = n_rep[0]
    if n_rep != K:
        raise ValueError(f"Number of repetitions computed from vec file ({n_rep})is not equal to the expected number of repetitions ({K}) originally used to generate the stimulus")

    # check the number of cycles
    n_cycles = total_frames / STIMULUS_FREQUENCY / (ON_DURATION + OFF_DURATION)

    FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_seconds = (ON_DURATION + OFF_DURATION) * len(frame_ids)
    FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_frames = int(FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_seconds * prediction_frequency)
    n_frames_onset = int(ON_DURATION * prediction_frequency)
    n_frames_offset = int(OFF_DURATION * prediction_frequency)
    n_frames_sequence = n_frames_onset + n_frames_offset

    disk_stimulus_processed = np.load(processed_spot_stack)

    # REPORT
    print(f"\nORIGINAL SPOT STIMULUS (Shown during experiment)\n"
          f"\tNumber of spot frames: {len(frames_sequence_no_rep)} (expected {len(frame_ids)})\n"
          f"\tNumber of times each frame is shown: {n_rep} (expected {K})\n"
          f"\tNumber of cycles: {n_cycles} (expected {len(frame_ids) * n_rep})\n"
          f"\tFrequency: {STIMULUS_FREQUENCY} Hz\n"
          f"\tDuration: {(ON_DURATION + OFF_DURATION) * len(frame_ids) * STIMULUS_FREQUENCY} frames (expected {total_frames/n_rep} = {vec_sequences[0, sequence_column]/n_rep})\n")
    print(f"\nFULL SPOT STIMULUS FOR MODELS (To be used as input for the models)\n"
          f"\tDuration: {FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_seconds} seconds ({FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_frames} frames)\n")

    # GENERATE FULL STIMULUS
    prediction_sequence = []
    prediction_stimulus = np.empty((FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_frames, disk_stimulus_processed.shape[1], disk_stimulus_processed.shape[2]))
    prediction_stimulus[:] = np.nan

    print("PRE-PROCESSING")
    print(f"Prediction stimulus shape: {prediction_stimulus.shape}")
    print(f"Number of nans: {np.sum(np.isnan(prediction_stimulus))} "
          f"(expected {FULL_SPOT_STIMULUS_FOR_MODELS_DURATION_frames * disk_stimulus_processed.shape[1] * disk_stimulus_processed.shape[2]})")

    black_screen_frame = disk_stimulus_processed[black_screen_frame_id]
    for i, frame_id in enumerate(frames_sequence_no_rep):
        # get the frame
        frame = disk_stimulus_processed[frame_id]

        # add the frame ids to the sequence
        prediction_sequence.extend([frame_id] * n_frames_onset)
        prediction_sequence.extend([black_screen_frame_id] * n_frames_offset)

        # add the frame to the stimulus
        i_onset = i * n_frames_sequence
        i_offset = i_onset + n_frames_onset
        end = i_offset + n_frames_offset
        prediction_stimulus[i_onset: i_offset] = frame
        prediction_stimulus[i_offset: end] = black_screen_frame

    np.save(full_spot_stimulus_for_models_frame_sequence, prediction_sequence)
    np.save(full_spot_stimulus_for_models, prediction_stimulus)

    print("AFTER-PROCESSING")
    print(f"Prediction stimulus shape: {prediction_stimulus.shape}")
    print(f"Number of nans: {np.sum(np.isnan(prediction_stimulus))} (expected 0)")
    print(f"Saved in {full_spot_stimulus_for_models}")




