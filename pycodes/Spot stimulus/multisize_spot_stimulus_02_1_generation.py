"""
Author : Chiara Boscarino
Created on : 2024.06.06


GENERATION of TILING MULTI-SIZE SPOT STIMULUS
This script enables the generation of the Multi-size spot stimulus for MEA recordings
once the design parameters have been established.

USER GUIDE
1. Choose what you want to compute;
2. Set the parameters in multisize_spot_stimulus_02_0_generation_parameters.py;
2. Run the codes to generate the stimulus;

"""

# 1. CHOOSE CODE TO RUN
# Generate set of coordinates for the spots,
# i.e. the locations of the spots (center) for each size
# to tile the target area
GENERATE_COORDINATES = True

# Generate the stack of spot frames for the stimulus,
# i.e. the npy file of dimension (tot_num_spots + 1, SPOT_STIMULUS_DIMENSION_px, SPOT_STIMULUS_DIMENSION_px)
# containing at [i,:,:] the frame of the i-th spot and the void frame in position 0
GENERATE_SPOT_FRAMES_NPY = True

# Generate the bin file of the stimulus,
# i.e. transform the npy in bin
GENERATE_SPOT_FRAMES_BIN = True

# Generate the vec file of the stimulus,
# i.e. define the sequence of frames to be shown during
# the experiment and save it in a vec file. This file will be used to build
# the stimulus online during the experiment showing the frames in the defined order,
# which is randomized.
GENERATE_SPOT_STIMULUS_VEC = True

# Generate the spot reference file,
# i.e. a file containing in the i-th row the info of the i-th frame of the stimulus
# starting from the original set of coordinates
GENERATE_SPOT_REFERENCE_FILE = True

# Visualize the spot arrangement,
# i.e. plot for each size all spots together on the target area,
# to visualize the tiling. Note that what will be actually shown is not the spot itself,
# but the max alignment margin area for each spot,
# i.e. MAX_ADMITTED_ALIGNMENT_MARGIN_PTG * spot_size,
# located in the center of the spot.
VISUALIZE_SPOT_ARRANGEMENT = True


# 2. SET PARAMETERS
# GO TO multisize_spot_stimulus_02_0_generation_parameters.py


# 3. CODE
from pycodes.modules.multisize_spot_stimulus_utils import *
from multisize_spot_stimulus_02_0_generation_parameters import *
from pycodes.modules import general_utils, visual_utils
import os

STIM_DIR = os.path.join(stimuli_dir, STIM_ID)
general_utils.make_dir(STIM_DIR)
coordinates_fp = os.path.join(STIM_DIR, coordinates_filename+".pkl")
frame_stack_fp = os.path.join(STIM_DIR, frame_stack_filename+".npy")
file_bin_fp = os.path.join(STIM_DIR, file_bin)
vec_fp = os.path.join(STIM_DIR, vec_filename+".vec")
spot_reference_fp = os.path.join(STIM_DIR, f"{spot_reference_filename}.vec")


# 3.1. GENERATE THE COORDINATES
if GENERATE_COORDINATES:

    print(f"\n\nGENERATING COORDINATES FOR {ARRANGEMENT} ARRANGEMENT WITH {MAX_ADMITTED_ALIGNMENT_MARGIN_PTG} ERROR\n")

    coords, tot_spots = simulate_stimulus(TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT, SIZES,
                                          MAX_ADMITTED_ALIGNMENT_MARGIN_PTG, ARRANGEMENT)

    print(f"\tTotal spots: {sum(tot_spots.values())}\n")
    for s, x in coords.items():
        print(f"\t{s}: {x.shape}")

    general_utils.save_dict(coords, coordinates_fp)

    # 2.2. VISUALIZE THE STIMULUS
    plot_stimulus(TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT, MAX_ADMITTED_ALIGNMENT_MARGIN_PTG, coords,
                  visualize_stimulation_area=True, target_area_shift=SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um,
                  total_area_width=SPOT_STIMULUS_DIMENSION_um, total_area_height=SPOT_STIMULUS_DIMENSION_um,
                  mea_size=MEA_SIZE_um)

    # ask the user if he wants to continue generating the stimulus
    ok_continue = input("\nDo you want to continue generating the stimulus? (y/n): ")
    if ok_continue.lower() != "y":
        print("Process interrupted by user")
        exit()

# Load coordinates
coordinates = general_utils.load_dict(coordinates_fp)
nb_spot_frames = sum([coordinates[disk_diameter].shape[0] for disk_diameter in coordinates.keys()])
nb_tot_frames = nb_spot_frames + 1

# 3.2. GENERATE NPY
if GENERATE_SPOT_FRAMES_NPY:
    print(f"\n\nGENERATING FRAME STACK (NPY)")

    print(f"Generating frame stack for {nb_spot_frames} spots ({nb_tot_frames} with the void frame)\n"
          f" Spots are displaced to tile a target area of {TARGET_AREA_WIDTH}x{TARGET_AREA_HEIGHT}µm\n"
          f" with a {MAX_ADMITTED_ALIGNMENT_MARGIN_PTG * 100}% error margin\n"
          f" The stimulus size is {SPOT_STIMULUS_DIMENSION_um}µm x {SPOT_STIMULUS_DIMENSION_um}µm ({SPOT_STIMULUS_DIMENSION_px}x{SPOT_STIMULUS_DIMENSION_px} pixels)\n"
          f" and spots will be shown starting from the top left corner of the mea at ({SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um},{SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um})µm.\n"
          f" (mea size: {MEA_SIZE_um}µm x {MEA_SIZE_um}µm, mea center: {SPOT_STIMULUS_CENTER_COORDINATE_um}µm)")

    frames = np.zeros((nb_tot_frames, SPOT_STIMULUS_DIMENSION_px, SPOT_STIMULUS_DIMENSION_px))

    # add the void frame in 0 position
    frames[0, :, :] = np.zeros((SPOT_STIMULUS_DIMENSION_px, SPOT_STIMULUS_DIMENSION_px))

    # add spot in the frame stack for each size and location sequentially
    i_frame = 1
    for disk_diameter in coordinates.keys():  # for each disk_size

        diameter_px = int(round(disk_diameter / PIXEL_SIZE))

        for i_disk, disk_coordinates in enumerate(coordinates[disk_diameter]):  # for each disk coordinate
            centered_coordinates = [SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um + i for i in disk_coordinates]
            centered_coordinates_px = (round(centered_coordinates[0] / PIXEL_SIZE),
                                       round(centered_coordinates[1] / PIXEL_SIZE))

            frames[i_frame, :, :] = create_disk(frame_dimensions,
                                                centered_coordinates_px,
                                                diameter_px,
                                                background_value=BACKGROUND_COLOR_VALUE,
                                                disk_value=SPOT_COLOR_VALUE)

            print(
                f"{i_frame}/{nb_tot_frames} frames generated: {disk_diameter}µm > ({int(round(centered_coordinates[0] / PIXEL_SIZE))},{int(round(centered_coordinates[1] / PIXEL_SIZE))})")
            i_frame += 1

    np.save(frame_stack_fp, frames)
    print(f" Stack shape: {frames.shape}\n"
          f" Saved file: {frame_stack_fp}")

# 3.3. GENERATE BIN
if GENERATE_SPOT_FRAMES_BIN:
    print(f"\n\nGENERATING FRAME STACK BIN")
    from pycodes.modules.binfile import BinFile

    frames = np.load(frame_stack_fp)

    bin_file = BinFile(file_bin_fp,
                       SPOT_STIMULUS_DIMENSION_px,
                       SPOT_STIMULUS_DIMENSION_px,
                       nb_images=nb_tot_frames,
                       rig_id=RIG_ID,
                       mode='w')

    # write frames
    for i_frame in range(frames.shape[0]):
        bin_file.append_frame(frames[i_frame, :, :])

    # Close file
    bin_file.close()
    print(f" Saved file: {file_bin_fp}")

# 3.4. GENERATE VEC
if GENERATE_SPOT_STIMULUS_VEC:

    stimulus_duration = INITIAL_ADAPTATION + (nb_spot_frames * (ON_DURATION + OFF_DURATION) * K)  # seconds
    nb_frames_displayed = stimulus_duration * STIMULUS_FREQUENCY

    print(f"\n\nGENERATING THE VEC FILE USING {K} REPETITIONS\n"
          f" Stimulus frequency: {STIMULUS_FREQUENCY} Hz\n"
          f" Stimulus duration: {stimulus_duration / 60} minutes\n"
          f" Tot. frames: {nb_frames_displayed}\n")

    # create empty array with an extra row for the header
    vec = np.empty((nb_frames_displayed + 1, 5))
    # fill the header
    vec[0, :] = [0, nb_frames_displayed, 0, 0, 0]

    # create the sequence of frames to be displayed (sequential)
    # (frames ranging from 1 to nb_spot_frames, +1 since last is excluded in ranges)
    repeated_sequence = np.tile(np.array(range(1, nb_spot_frames + 1)), K)

    # randomize the sequence
    np.random.seed(0)
    randomized_repeated_sequences = np.random.permutation(repeated_sequence)

    sequences_to_vec = []
    frames_to_vec = []

    # Add grey adaptation frames in the beginning
    sequences_to_vec += INITIAL_ADAPTATION * STIMULUS_FREQUENCY * [0]
    frames_to_vec += INITIAL_ADAPTATION * STIMULUS_FREQUENCY * [0]

    # Add spots frames
    for sequence in randomized_repeated_sequences:
        sequences_to_vec += (ON_DURATION + OFF_DURATION) * STIMULUS_FREQUENCY * [sequence]
        frames_to_vec += ON_DURATION * STIMULUS_FREQUENCY * [sequence]
        frames_to_vec += OFF_DURATION * STIMULUS_FREQUENCY * [0]

    vec[1:, :] = 0
    vec[1:, 1] = frames_to_vec
    vec[1:, 4] = sequences_to_vec

    # write the vec array in a .vec file
    with open(vec_fp, "w") as f:
        np.savetxt(f, vec, delimiter=',', fmt='%i %i %i %i %i')
    print(f" Saved file: {vec_fp}")

# 3.5. GENERATE REFERENCE FILE
if GENERATE_SPOT_REFERENCE_FILE:

    print(f"\n\nGENERATING SPOT REFERENCE FILE")
    expected_tot_spots = sum([x.shape[0] for x in coordinates.values()])
    expected_sizes = list(coordinates.keys())
    print(f" Total spots: {expected_tot_spots}\n"
          f" Spot sizes: {expected_sizes}\n")

    # The before defined coordinates are defined to tile an independent stimulation area
    # whose reference frame has the (0,0) in the upper left corner.
    # Yet, the spots have been located in a reduced area aligned with the mea
    # (real coordinates are original coordinates + (stimulus_center - MEA_SIZE_um / 2),
    # considering top left corner of the stimulus as (0,0)) --> transformed coordinates.
    # Also, we want to know the spot coordinates with respect to the center of the stimulus,
    # so to be able to use the same reference frame for cell and spot location --> aligned coordinates.
    # Thus, we want to store for each spot (identified by the reference frame number)
    # size and coordinates, both original, transformed and aligned.

    spot_reference_header = ("size "
                             "X_original "
                             "Y_original "
                             "X "
                             "Y "
                             "X_aligned "
                             "Y_aligned")

    # Add the row 0 for the frame 0 (black screen)
    sizes = [0]
    Xs_original = [0]
    Ys_original = [0]
    Xs = [0]
    Ys = [0]
    Xs_aligned = [0]
    Ys_aligned = [0]

    # Then for each spot envisaged in the coordinates file add the row for the frame
    # Since the frames have been generated following the coords order
    # we can use the index of the spot in the list of coords as frame id,
    # --> spot_reference_file[frame_id] = spot_info
    # --> stimulus_npy[frame_id,:,:] = stimulus_frame
    for diameter, list_of_coords in coordinates.items():
        for i, coords in enumerate(list_of_coords):
            x = coords[0]
            y = coords[1]
            actual_size, actual_x, actual_y = transform_info(diameter, x, y,
                                                             SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um)
            x_aligned, y_aligned = general_utils.align_coordinates(actual_x, actual_y,
                                                                   SPOT_STIMULUS_CENTER_COORDINATE_um,
                                                                   SPOT_STIMULUS_CENTER_COORDINATE_um)

            sizes += [actual_size]
            Xs_original += [x]
            Ys_original += [y]
            Xs += [actual_x]
            Ys += [actual_y]
            Xs_aligned += [x_aligned]
            Ys_aligned += [y_aligned]

    # Store the table
    table = np.array([sizes, Xs_original, Ys_original, Xs, Ys, Xs_aligned, Ys_aligned]).T
    general_utils.array_to_vec_file(table, header=spot_reference_header, filename=spot_reference_fp)
    print(f" Stored table: {table.shape}")
    print(f" Saved file: {spot_reference_fp}")

    # Reload the file and check the content
    frames_reference = np.genfromtxt(spot_reference_fp)
    nb_frames = frames_reference.shape[0]
    spot_sizes = np.unique(frames_reference[:, 0])
    spot_sizes = np.delete(spot_sizes, 0)  # remove the 0 (black screen)
    print(f"\nREFERENCE TABLE\n"
          f"\tNumber of frames (included black screen): {nb_frames} (expected {expected_tot_spots + 1})\n"
          f"\tHeader: {spot_reference_header}\n"
          f"\tSpot sizes: {len(spot_sizes)} ({spot_sizes})\n\n"
          f"\tFrame 0: {get_frame_info(frames_reference, 0)}\n"
          f"\tFrame 1: {get_frame_info(frames_reference, 1)}\n"
          "\t...\n"
          f"\tFrame {expected_tot_spots-1}: {get_frame_info(frames_reference, expected_tot_spots-1)}\n"
          f"\n\tTable saved in {spot_reference_fp}")

    # check 1 - NB FRAMES == NB SPOTS + 1 (void frame)
    if frames_reference.shape[0] != expected_tot_spots + 1:
        print(f"WARNING: number of frames in the reference file is "
              f"{frames_reference.shape[0]} (expected {expected_tot_spots + 1})")

    # check 2 - NB SPOT SIZES == SPOT SIZES IN COORDINATES FILE + 0 (void frame)
    if set(spot_sizes) != set(expected_sizes):
        print(f"WARNING: spot sizes in the reference file are "
              f"{spot_sizes} (expected {expected_sizes})")

    # check 3 - CORRESPONDENCE BETWEEN FRAME IN NPY AND INFO IN REFERENCE FILE
    # select a set of frames to check
    frames_to_check = np.random.choice(range(0, expected_tot_spots), 12, replace=False)

    print(f"\n\nCHECK CORRESPONDENCE BETWEEN FRAME IN NPY AND INFO IN REFERENCE FILE\n"
          f"\tSize: {SPOT_STIMULUS_DIMENSION_um} µm ({SPOT_STIMULUS_DIMENSION_px} pixels)\n"
          f"\tOrigin: {(TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT)} µm ({tuple((np.array([TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT])/PIXEL_SIZE).astype(int))} pixels)\n"
          f"\tMEA size: {MEA_SIZE_um} µm ({int(MEA_SIZE_um / PIXEL_SIZE)} pixels)\n")

    if GENERATE_SPOT_FRAMES_NPY or GENERATE_SPOT_FRAMES_BIN:
        disk_stimulus = frames
    else:
        print(f"Loading frame stack from {frame_stack_fp}")
        disk_stimulus = np.load(frame_stack_fp)

    ncols = 3
    nrows = int(np.ceil(len(frames_to_check) / ncols))
    imdim = 5
    fig = plt.figure(figsize=(imdim * ncols, imdim * nrows))
    for i, fid in enumerate(frames_to_check):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        visual_utils.add_spot_frame_to_figure(fid,
                                              disk_stimulus, frames_reference,
                                              SPOT_STIMULUS_DIMENSION_um, MEA_SIZE_um,
                                              ax, grid=True)
        ax.set_title(f"Frame {fid}")
        info = get_frame_info(frames_reference, fid)
        info_text = (f"sizes = {info[0]}\n"
                     f"orig = ({info[1]}, {info[2]})\n"
                     f"transf = ({info[3]}, {info[4]})\n"
                     f"aligned = ({info[5]}, {info[6]})\n")
        ax.text(-SPOT_STIMULUS_DIMENSION_um/2, SPOT_STIMULUS_DIMENSION_um/2, info_text,
                color='black', fontsize=10, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=1))
    plt.show()

# 3.6. VISUALIZE SPOT ARRANGEMENT
if VISUALIZE_SPOT_ARRANGEMENT:

    print(f"\n\nVISUALIZING SPOT ARRANGEMENT")
    frames_reference = np.genfromtxt(spot_reference_fp)

    # Retrieve from the reference file the frames organized per spot size
    # (iterate by rows the reference frame, retrieve for each row the size
    # and add the frame id to the list of frames for that size)
    frames_id_per_size = {}
    for frame_id in range(frames_reference.shape[0]):
        size, _, _, _, _, _, _ = get_frame_info(frames_reference, frame_id)
        if size not in frames_id_per_size.keys():
            frames_id_per_size[size] = []
        frames_id_per_size[size] += [frame_id]

    print(f"\tSizes: {len(frames_id_per_size.keys())} ({list(frames_id_per_size.keys())})\n"
          f"\tFrames per size: {np.unique([len(frames_id_per_size[size]) for size in frames_id_per_size.keys()])}\n"
          f"\tTotal frames: {sum([len(frames_id_per_size[size]) for size in frames_id_per_size.keys()])}\n")

    # Plot the tiling arrangement of the frames per size
    for size in frames_id_per_size.keys():
        fig = plt.figure(figsize=(20, 10))
        lw = max(2, size / 200)
        fig.suptitle(f"Size {size} µm ({len(frames_id_per_size[size])} frames)")

        # CENTERED um
        ax1 = fig.add_subplot(121)
        dim = int(SPOT_STIMULUS_DIMENSION_um)
        background = np.zeros((dim, dim))
        # ax1.grid()
        ax1.imshow(background, cmap='gray', extent=(-SPOT_STIMULUS_DIMENSION_um/2, SPOT_STIMULUS_DIMENSION_um/2, -SPOT_STIMULUS_DIMENSION_um/2, SPOT_STIMULUS_DIMENSION_um/2))
        ax1.add_patch(plt.Rectangle((-MEA_SIZE_um/2, MEA_SIZE_um/2), MEA_SIZE_um, -MEA_SIZE_um, edgecolor='gray', fill=False))
        ax1.scatter(0, 0, color='r')
        ax1.set_title(f"Whole stimulation area centered (µm)\nMEA in gray ({MEA_SIZE_um} µm)")

        # ORIGINAL
        ax2 = fig.add_subplot(122)
        background = np.zeros((int(TARGET_AREA_WIDTH), int(TARGET_AREA_HEIGHT)))
        # ax2.grid()
        ax2.imshow(background, cmap='gray',
                   extent=(0, TARGET_AREA_WIDTH, TARGET_AREA_HEIGHT, 0))
        ax2.scatter(TARGET_AREA_WIDTH/2, TARGET_AREA_HEIGHT/2, color='r')
        ax2.set_title(f"Spot target area ({TARGET_AREA_WIDTH}x{TARGET_AREA_HEIGHT} µm)")
        ax2.set_axis_off()

        for idx_f, frame_id in enumerate(frames_id_per_size[size]):
            size, x_original, y_original, x_actual, y_actual, x_aligned, y_aligned = get_frame_info(frames_reference, frame_id)
            # CENTERED um
            ax1.scatter(x_aligned, y_aligned, color='gray', s=10)
            if idx_f in [0, 1]: ax1.add_patch(plt.Circle((x_aligned, y_aligned), size/2, edgecolor='yellow', fill=False))
            ax1.add_patch(plt.Circle((x_aligned, y_aligned), radius=MAX_ADMITTED_ALIGNMENT_MARGIN_PTG * size/2, edgecolor='green', fill=False))
            # ORIGINAL
            ax2.scatter(x_original, y_original, color='gray')
            if idx_f in [0, 1]: ax2.add_patch(plt.Circle((x_original, y_original), size/2, edgecolor='yellow', linewidth=lw*0.6, fill=False))
            ax2.add_patch(plt.Circle((x_original, y_original), radius=MAX_ADMITTED_ALIGNMENT_MARGIN_PTG * size / 2, edgecolor='green', linewidth=lw, fill=False))
        plt.show()
