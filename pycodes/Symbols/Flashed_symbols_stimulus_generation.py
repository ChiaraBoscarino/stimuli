import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pycodes.modules import general_utils, symbol_stimuli_utils

# ---------------------------------------------------------------------------------------------- #
# >> CODE MODULATORS
GENERATE_NPY = True
GENERATE_BIN = True
GENERATE_VEC = True
VISUALIZE_STIMULUS_GIF = True
# ---------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------- #
# >> PARAMETERS
rootpath = "C:\\Users\\chiar\\Documents\\stimuli"
output_root_folder = os.path.join(rootpath, "FlashedSymbols")

set_of_params = {
    "STIMULUS_VERSION_ID": "FlashedSymbols_67_300ms",
    "RIG_ID": 1,
    "mea_size": 1530,  # in µm

    "pixel_size": 3.5,  # in µm per pixel
    "stimulus_frequency": 15,  # Hz

    "SA_sq_dim": 768,  # pixels

    "symbols": ['F', 'T', 'I'],
    "symbol_sizes_um": [150, 300, 450],  # µm
    "symbol_locations": {"150um": [(0.4, 0.4), (0.4, 0.6), (0.6, 0.4), (0.6, 0.6)],
                         "300um": [(0.4, 0.4), (0.6, 0.6)],
                         "450um": [(0.5, 0.5)]},
    "onset_time": 0.067,  # seconds
    "offset_time": 0.300,  # seconds
    "initial_adaptation": 1,  # seconds

    "symbol_color": 1,
    "background_color": 0,

    "n_repetitions": 20
}

# ---------------------------------------------------------------------------------------------- #


def main():
    parameters = set_of_params
    Stimulus_ID = parameters["STIMULUS_VERSION_ID"]
    RIG_ID = parameters["RIG_ID"]
    mea_size = parameters["mea_size"]
    pixel_size = parameters["pixel_size"]
    stimulus_frequency = parameters["stimulus_frequency"]
    SA_sq_dim = parameters["SA_sq_dim"]
    symbols = parameters["symbols"]
    symbol_sizes_um = parameters["symbol_sizes_um"]
    symbol_locations = parameters["symbol_locations"]
    onset_time = parameters["onset_time"]
    offset_time = parameters["offset_time"]
    initial_adaptation = parameters["initial_adaptation"]
    symbol_color = parameters["symbol_color"]
    background_color = parameters["background_color"]
    nreps = parameters["n_repetitions"]

    dt = 1 / stimulus_frequency  # timestep of a frame in seconds
    onset_time_frames = int(onset_time * stimulus_frequency)  # frames
    offset_time_frames = int(offset_time * stimulus_frequency)  # frames
    initial_adaptation_frames = int(initial_adaptation * stimulus_frequency)  # frames
    single_sequence_duration = onset_time + offset_time  # seconds
    single_sequence_duration_frames = onset_time_frames + offset_time_frames  # frames

    SA_x_size_px, SA_y_size_px = SA_sq_dim, SA_sq_dim  # pixels
    SA_x_size_um, SA_y_size_um = SA_x_size_px * pixel_size, SA_y_size_px * pixel_size  # micrometers
    parameters["SA_x_size_um"], parameters["SA_y_size_um"] = SA_x_size_um, SA_y_size_um

    symbol_locations_px = {k: [(int(x*SA_x_size_px), int(y*SA_y_size_px)) for (x, y) in locs] for k, locs in symbol_locations.items()}
    symbol_locations_um = {k: [(x*pixel_size, y*pixel_size) for (x,y) in locs] for k, locs in symbol_locations_px.items()}
    symbol_sizes_px = np.array([int(x // pixel_size) for x in symbol_sizes_um])  # pixels
    n_sequences_per_symbol = np.sum([len(locs) for _, locs in symbol_locations.items()])
    n_sequences = len(symbols) * n_sequences_per_symbol
    tot_seqs = n_sequences * nreps
    tot_dur = initial_adaptation + tot_seqs * single_sequence_duration  # seconds
    tot_n_frames = initial_adaptation_frames + tot_seqs * single_sequence_duration_frames  # frames

    output_folder = os.path.join(output_root_folder, Stimulus_ID)
    general_utils.make_dir(output_folder)
    param_path = os.path.join(output_folder, f"{Stimulus_ID}_parameters.json")
    frame_stack_fp = os.path.join(output_folder, f"{Stimulus_ID}_frame_stack.npy")
    file_bin_fp = os.path.join(output_folder, f"{Stimulus_ID}_frame_stack.bin")
    reference_table_fp = os.path.join(output_folder, f"{Stimulus_ID}_reference_table.csv")
    vec_fp = os.path.join(output_folder, f"{Stimulus_ID}_vec.vec")
    general_utils.write_json(parameters, param_path)

    # ---------------------------------------------------------------------------------------------- #
    # >> CODE
    # - Visualize locations
    SA_center_x, SA_center_y = SA_x_size_um / 2, SA_y_size_um / 2
    nrows = 1
    ncols = len(symbol_locations_um)
    dim = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * dim, nrows * dim))
    for i, (size, locs) in enumerate(symbol_locations_um.items()):
        ax = axs[i]
        ssum = int(size[:-2])
        for loc in locs:
            ax.plot(loc[0], loc[1], 'o', markersize=5, color='black')
            ax.add_patch(plt.Rectangle((loc[0] - ssum / 2, loc[1] - ssum / 2),
                                       ssum, ssum, fill=None, edgecolor='black'))
        ax.set_title(f"{size}")
        ax.set_xlim(0, SA_x_size_um)
        ax.set_ylim(0, SA_y_size_um)
        ax.add_patch(plt.Rectangle((SA_center_x-mea_size/2, SA_center_y-mea_size/2), mea_size, mea_size, alpha=0.2))
    plt.tight_layout()
    plt.show()
    general_utils.save_figure(fig, f"{Stimulus_ID}_symbol_locations.jpg", output_folder)

    # - Summary
    print(f"\n\nGENERATING STIMULUS: {Stimulus_ID}")
    print(f" - Pixel size: {pixel_size} µm/px")
    print(f" - Stimulation area: {SA_x_size_um}x{SA_y_size_um} µm ({SA_x_size_px}x{SA_y_size_px} px)")
    print(f" - Symbols: {symbols}")
    print(f" - Symbol sizes: {symbol_sizes_um} µm")
    print(f" - Symbol locations: {symbol_locations_um}")
    print(f" - Stimulus frequency: {stimulus_frequency} Hz (dt: {dt} s)")
    print(f" - Onset time: {onset_time} s, offset time: {offset_time} s ({onset_time_frames}, {offset_time_frames} frames)")
    print(f" - N sequences: {n_sequences} ({nreps} repetitions --> {tot_seqs} sequences)")
    print(f" - Tot duration: {tot_dur//60}' {tot_dur%60} s ({tot_n_frames} frames of which {initial_adaptation_frames} of initial adaptation)")
    print(f" - Symbol color: {symbol_color} (background: {background_color})")
    print()
    print(f"Code to run: "
          f"GENERATE_NPY={GENERATE_NPY}, GENERATE_BIN={GENERATE_BIN}, "
          f"GENERATE_VEC={GENERATE_VEC}, VISUALIZE_STIMULUS_GIF={VISUALIZE_STIMULUS_GIF}")
    # ask user to confirm
    ok = input("\nContinue? (y/n): ")
    if ok.lower() != "y":
        print("Aborted.")
        return

    # - GENERATE NPY
    if GENERATE_NPY:
        print(f"\n\nGENERATING FRAME STACK (NPY)")

        # Initialize the frame stack
        flash_frame_stack = np.zeros((n_sequences+1, SA_y_size_px, SA_x_size_px))
        # add the void frame in 0 position
        flash_frame_stack[0, :, :] = np.zeros((SA_y_size_px, SA_x_size_px))

        # Initialize the reference table
        reference_table = {'SID': [], 'SYMBOL': [], 'SIZE (µm)': [], 'LOCATION (µm)': []}

        sid = 1
        for symbol in symbols:
            for symbol_size_um, symbol_size_px in zip(symbol_sizes_um, symbol_sizes_px):
                symbol_loc_px = symbol_locations_px[f"{symbol_size_um}um"]
                symbol_loc_um = symbol_locations_um[f"{symbol_size_um}um"]
                for loc_um, loc_px in zip(symbol_loc_um, symbol_loc_px):

                    # Generate flash frame
                    flash_frame_stack[sid, :, :] = symbol_stimuli_utils.generate_symbol_frame(symbol,
                                                                                              SA_x_size_px, SA_y_size_px,
                                                                                              symbol_size_px,
                                                                                              loc_px,
                                                                                              symbol_color, background_color)

                    # Update reference table
                    reference_table['SID'].append(sid)
                    reference_table['SYMBOL'].append(symbol)
                    reference_table['SIZE (µm)'].append(symbol_size_um)
                    reference_table['LOCATION (µm)'].append(loc_um)

                    sid += 1

        # Check
        assert sid == n_sequences+1, f"Error: {sid} != {n_sequences}"
        assert len(reference_table['SID']) == n_sequences, f"Error: {len(reference_table['SID'])} != {n_sequences}"
        assert flash_frame_stack.shape[0] == n_sequences+1, f"Error: {flash_frame_stack.shape[0]} != {n_sequences}"

        # Save the frame stack
        np.save(frame_stack_fp, flash_frame_stack)
        print(f" Flash frame stack: {flash_frame_stack.shape}\n"
              f" --> {frame_stack_fp}\n")

        # Save the reference table
        reference_table = pd.DataFrame(reference_table)
        reference_table.to_csv(reference_table_fp, index=False)
        print(f" Reference table: {n_sequences} sequences x ({list(reference_table.columns)})")
        print(f" --> {reference_table_fp}")

    if GENERATE_BIN:
        print(f"\n\nSTORING FRAME STACK (BIN)")
        from pycodes.modules.binfile import BinFile

        frames = np.load(frame_stack_fp)

        bin_file = BinFile(file_bin_fp,
                           SA_y_size_px, SA_x_size_px,
                           nb_images=len(frames),
                           rig_id=RIG_ID,
                           mode='w')

        # write frames
        for i_frame in range(frames.shape[0]):
            bin_file.append_frame(frames[i_frame, :, :])

        # Close file
        bin_file.close()
        print(f" --> {file_bin_fp}")

    if GENERATE_VEC:
        print(f"\n\nGENERATING VEC FILE")

        repeated_sequence = np.tile(np.array(range(1, n_sequences + 1)), nreps)
        assert len(repeated_sequence) == tot_seqs, f"Error: {len(repeated_sequence)} != {tot_seqs}"

        # randomize the sequence
        np.random.seed(0)
        randomized_repeated_sequences = np.random.permutation(repeated_sequence)

        # Build vectors
        frames_to_vec = []
        sequences_to_vec = []

        # Initial adaptation
        sequences_to_vec += initial_adaptation_frames * [0]
        frames_to_vec += initial_adaptation_frames * [0]

        # Flash frames
        for sequence in randomized_repeated_sequences:
            sequences_to_vec += (onset_time_frames + offset_time_frames) * [sequence]
            frames_to_vec += (onset_time_frames * [sequence]) + (offset_time_frames * [0])

        assert len(sequences_to_vec) == len(frames_to_vec), f"Error: {len(sequences_to_vec)} != {len(frames_to_vec)}"
        assert len(sequences_to_vec) == tot_n_frames, f"Error: {len(sequences_to_vec)} != {tot_n_frames}"

        # vec table generation
        vec = np.empty((tot_n_frames + 1, 5))  # create empty array with an extra row for the header
        vec[0, :] = [0, tot_n_frames, 0, 0, 0]  # fill the header
        vec[1:, :] = 0  # fill the rest of the array with zeros
        vec[1:, 1] = frames_to_vec  # fill the frames column (exact frame nb to show)
        vec[1:, 4] = sequences_to_vec  # fill the sequences column (ongoing sequence nb)

        # vec table to .vec file
        with open(vec_fp, "w") as f:
            np.savetxt(f, vec, delimiter=',', fmt='%i %i %i %i %i')
        print(f"Vec table: {vec.shape}")
        print(f"--> {vec_fp}")

    if VISUALIZE_STIMULUS_GIF:
        print(f"\n\nVISUALIZING STIMULUS GIF")
        from pycodes.modules import gif

        frame_stack = np.load(frame_stack_fp)
        vec = np.genfromtxt(vec_fp)
        frames_to_vec = vec[1:, 1]

        complete_stack_frame = np.array([frame_stack[int(f), :, :] for f in frames_to_vec])

        assert complete_stack_frame.shape[0] == tot_n_frames, f"Error: {complete_stack_frame.shape[0]} != {tot_n_frames}"

        gif_fp = os.path.join(output_folder, f"{Stimulus_ID}.gif")
        gif.create_gif(complete_stack_frame, gif_fp, dt=dt * 1000, loop=1)
        print(f"Gif: {complete_stack_frame.shape} frames")
        print(f" --> {gif_fp}")

    return


if __name__ == "__main__":
    main()
