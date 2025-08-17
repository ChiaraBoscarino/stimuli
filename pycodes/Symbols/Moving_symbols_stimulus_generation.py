import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pycodes.modules import general_utils, symbol_stimuli_utils
from pycodes.modules import gif

# ---------------------------------------------------------------------------------------------- #
# >> CODE MODULATORS
VISUALIZE_TRAJECTORIES = True
GENERATE_NPY = True
STORE_SINGLE_SEQUENCE_GIF = True  # (to run this GENERATE_NPY = True)
GENERATE_BIN = False
GENERATE_VEC = False
VISUALIZE_STIMULUS_GIF = False
# ---------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------- #
# >> PARAMETERS
rootpath = "C:\\Users\\cboscarino\\Documents\\GitHub\\stimuli"
# rootpath = "D:\\STIMULI"
output_root_folder = os.path.join(rootpath, "MovingSymbols")
general_utils.make_dir(output_root_folder)

set_of_params = {
    "STIMULUS_VERSION_ID": "MovSyb_for_ppt",
    "RIG_ID": 1,
    "mea_size": 1530,  # in µm

    "pixel_size": 3.5,  # in µm per pixel
    "stimulus_frequency": 40,  # Hz

    "SA_sq_dim": 100,  # pixels

    # - Symbols
    "symbols": ['E'],
    "symbol_sizes_um": [50, 150],  # µm
    "symbol_color": 1,
    "background_color": 0,

    # - Movement
    # Trajectories are lists of (x, y) coordinates in the range [0, 1]
    # corresponding to the normalized position of the symbol center on the screen.
    "symbol_trajectories": {
                # "Trj1": [(0.5, 0.5), (0.6, 0.3), (0.5, 0.5), (0.6, 0.6), (0.3, 0.7), (0.3, 0.3)],
                # "Horizontal": [(0.15, 0.37), (0.85, 0.37)],
                # "DiagonalBL2TR": [(0.2, 0.8), (0.8, 0.2)],
                # "DiagonalTL2BR": [(0.2, 0.2), (0.8, 0.8)],
                # "Horizontal_TOP_L2R": [(0.15, 0.37), (0.85, 0.37)],
                # "Horizontal_CENTER_L2R": [(0.15, 0.5), (0.85, 0.5)],
                # "Horizontal_BOTTOM_L2R": [(0.15, 0.63), (0.85, 0.63)],
                "Random": [(0.25, 0.25), (0.6, 0.3),(0.75, 0.63), (0.5, 0.6), (0.25, 0.7), (0.75,0.75), (0.25, 0.25), (0.6, 0.3),(0.75, 0.63), (0.5, 0.6), (0.25, 0.7), (0.75,0.75)],
                },
    "symbol_speeds": [50],  # pixels/s
    "fixation_time": 0,  # seconds
    "s2s_transition_time": 0,  # seconds  # KEEP TO 0 - NOT IMPLEMENTED IN THE CODE

    "initial_adaptation": 0,  # seconds

    "n_repetitions": 20
}

# ---------------------------------------------------------------------------------------------- #


def main():

    # ---------------------------------------------------------------------------------------------- #
    parameters = set_of_params
    parameters["STIMULUS_VERSION_ID"] = f"{parameters['STIMULUS_VERSION_ID']}_{parameters['stimulus_frequency']}Hz"

    Stimulus_ID = parameters["STIMULUS_VERSION_ID"]
    RIG_ID = parameters["RIG_ID"]
    mea_size = parameters["mea_size"]
    pixel_size = parameters["pixel_size"]
    stimulus_frequency = parameters["stimulus_frequency"]
    SA_sq_dim = parameters["SA_sq_dim"]
    symbols = parameters["symbols"]
    symbol_sizes_um = parameters["symbol_sizes_um"]
    symbol_color = parameters["symbol_color"]
    background_color = parameters["background_color"]
    symbol_trajectories = parameters["symbol_trajectories"]
    symbol_speeds = parameters["symbol_speeds"]
    fixation_time = parameters["fixation_time"]
    s2s_transition_time = parameters["s2s_transition_time"]
    initial_adaptation = parameters["initial_adaptation"]
    nreps = parameters["n_repetitions"]

    dt = 1 / stimulus_frequency  # timestep of a frame in seconds

    SA_x_size_px, SA_y_size_px = SA_sq_dim, SA_sq_dim  # pixels
    SA_x_size_um, SA_y_size_um = SA_x_size_px * pixel_size, SA_y_size_px * pixel_size  # µm
    SA_center_x, SA_center_y = SA_x_size_px // 2, SA_y_size_px // 2  # pixels
    mea_size_px = int(mea_size // pixel_size)  # pixels
    parameters["SA_x_size_um"], parameters["SA_y_size_um"] = SA_x_size_um, SA_y_size_um

    symbol_sizes_px = [int(x // pixel_size) for x in symbol_sizes_um]  # pixels
    symbol_speeds_um = [int(x * pixel_size) for x in symbol_speeds]  # µm/s

    initial_adaptation_frames = int(initial_adaptation * stimulus_frequency)  # frames
    fixation_time_frames = int(fixation_time * stimulus_frequency)  # frames
    s2s_transition_frames = int(s2s_transition_time * stimulus_frequency)  # frames

    n_sequences_per_symbol = len(symbol_trajectories) * len(symbol_speeds) * len(symbol_sizes_um)
    n_sequences = len(symbols) * n_sequences_per_symbol
    tot_seqs = n_sequences * nreps

    output_folder = os.path.join(output_root_folder, Stimulus_ID)
    param_path = os.path.join(output_folder, f"{Stimulus_ID}_parameters.json")
    files_folder = os.path.join(output_folder, "files")
    trajectories_fp = os.path.join(output_folder, f"{Stimulus_ID}_trajectories.pkl")
    bin_frame_stack_fp = os.path.join(output_folder, f"{Stimulus_ID}_bin_frame_stack.npy")
    file_bin_fp = os.path.join(output_folder, f"{Stimulus_ID}_frame_stack.bin")
    reference_table_fp = os.path.join(output_folder, f"{Stimulus_ID}_reference_table.csv")
    vec_fp = os.path.join(output_folder, f"{Stimulus_ID}_vec.vec")

    if 0 <= symbol_color < 255 and 0 <= background_color < 255:
        dtype = np.uint8
    else:
        dtype = np.int8

    # ---------------------------------------------------------------------------------------------- #

    # -- Compute trajectory point sequence
    symbol_center_trajectory = {}
    for trj_name, trajectory in symbol_trajectories.items():
        trj = []
        for (x, y) in trajectory:
            # check trj point in [0, 1]
            if x < 0 or x > 1 or y < 0 or y > 1: raise ValueError("Trajectory coordinates must be between 0 and 1")
            # compute trj point in pixels on the SA
            trj_point_px = (int(x*SA_x_size_px), int(y*SA_y_size_px))
            # check trj point in SA for all symbol sizes
            for symbol_size_px in symbol_sizes_px:
                symbol_stimuli_utils.symbol_location_check(trj_point_px, symbol_size_px, SA_x_size_px, SA_y_size_px)
            # append trj point to the trajectory
            trj.append(trj_point_px)
        # append the trajectory to the list of trajectories
        symbol_center_trajectory[trj_name] = trj

    # -- Show trajectories
    fig_traj = None
    if VISUALIZE_TRAJECTORIES:
        ncols = 2
        nrows = len(symbol_center_trajectory)
        dim = 3
        fontsize = 12
        label_fontsize = 9
        fig = plt.figure(figsize=(ncols * dim, nrows * dim))
        gs = fig.add_gridspec(nrows, ncols)
        for i_t, (trj_name, trajectory) in enumerate(symbol_center_trajectory.items()):
            # - trajectory path
            ax = fig.add_subplot(gs[i_t, 0])
            ax.set_xlim(0, SA_x_size_px)
            ax.set_ylim(0, SA_y_size_px)
            ax.add_patch(plt.Rectangle((SA_center_x - mea_size_px / 2, SA_center_y - mea_size_px / 2), mea_size_px, mea_size_px, alpha=0.2))
            ax.invert_yaxis()  # invert y-axis to have (0, 0) in the top left corner
            x, y = zip(*trajectory)
            ax.plot(x, y, '-o', linewidth=1, markersize=17, markerfacecolor='white', color='k')
            for i_tp, trj_pt in enumerate(trajectory): ax.text(trj_pt[0], trj_pt[1], f"{i_tp}", fontsize=label_fontsize,
                                                               ha='center', va='center')
            ax.set_title(f"{trj_name}", fontsize=fontsize)
            ax.set_xlabel("x (px)", fontsize=label_fontsize)
            ax.set_ylabel("y (px)", fontsize=label_fontsize)

            # - symbol sizes
            ax = fig.add_subplot(gs[i_t, 1])
            ax.set_xlim(0, SA_x_size_px)
            ax.set_ylim(0, SA_y_size_px)
            ax.add_patch(plt.Rectangle((SA_center_x - mea_size_px / 2, SA_center_y - mea_size_px / 2), mea_size_px, mea_size_px, alpha=0.2))
            ax.invert_yaxis()
            x, y = zip(*trajectory)
            ax.scatter(x, y, s=5, color='k')
            for i_s, symbol_size_px in enumerate(symbol_sizes_px):
                for i_tp, trj_pt in enumerate(trajectory):
                    ax.add_patch(plt.Rectangle((trj_pt[0] - symbol_size_px / 2, trj_pt[1] - symbol_size_px / 2), symbol_size_px, symbol_size_px, color='k', linewidth=1, fill=False))

        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()
        fig_traj = fig

    # -- Compute full trajectories
    full_trajectories = {}
    for speed in symbol_speeds:
        full_trajectories[speed] = {}
        for trj_name, trajectory in symbol_center_trajectory.items():
            ft = []
            for i_tp, trj_point in enumerate(trajectory):
                # fixation
                fixation = [[trj_point[0], trj_point[1]] for _ in range(fixation_time_frames)]
                ft.extend(fixation)

                # if last trj point stop after fixation
                # else add point to point transition
                if i_tp == len(trajectory)-1: continue
                start_pt, end_pt = np.array(trj_point), np.array(trajectory[i_tp+1])
                distance = np.linalg.norm(end_pt - start_pt)  # in pixels
                nb_frames = int(distance / speed * stimulus_frequency)
                step_x, step_y = (end_pt[0] - start_pt[0]) / nb_frames, (end_pt[1] - start_pt[1]) / nb_frames
                transition = [[int(start_pt[0]+i*step_x), int(start_pt[1]+i*step_y)] for i in range(nb_frames)]
                ft.extend(transition)
            full_trajectories[speed][trj_name] = np.array(ft)

    # -- Trajectories duration
    trj_duration = {}
    for speed, tjx in full_trajectories.items():
        trj_duration[speed] = {n: ft.shape[0]/stimulus_frequency for n, ft in tjx.items()}
    max_trj_duration_frames = np.max([int(np.max(list(tjx_durations.values())) * stimulus_frequency) for tjx_durations in trj_duration.values()])

    # -- Total stimulus duration
    tot_stim_duration_s = (np.sum([np.sum(np.array(list(tjx_durations.values())) + s2s_transition_time) for speed, tjx_durations in trj_duration.items()])
                           * len(symbols) * len(symbol_sizes_um) * nreps + initial_adaptation)
    tot_stim_duration_frames = int(np.round(tot_stim_duration_s * stimulus_frequency))

    # - User confirmation
    report_string = (f"\n>> Generating Moving Symbols stimulus (Version ID {Stimulus_ID})\n"
                     f"\t- Stimulus frequency: {stimulus_frequency} Hz\n"
                     f"\t- Frame size: {SA_x_size_um}x{SA_y_size_um} µm ({SA_x_size_px}x{SA_y_size_px} px)\n"
                     f"\n"
                     f"\t- Symbols: {symbols}\n"
                     f"\t- Symbol sizes: {symbol_sizes_um} µm ({symbol_sizes_px} px)\n"
                     f"\n"
                     f"\t- Symbol trajectories: {len(symbol_center_trajectory)} ({np.unique([len(x) for x in symbol_center_trajectory.values()])} points)\n"
                     f"\t- Symbol speeds: {symbol_speeds_um} µm/s ({symbol_speeds} px/s)\n"
                     f"\t- Fixation time: {fixation_time} s ({fixation_time_frames} frames)\n"
                     f"\t- Trajectories duration: {trj_duration} [speed (px/s): trj_x_dur (s)]\n"
                     f"\t- Transition time between sequences: {s2s_transition_time} s ({s2s_transition_frames} frames)\n"
                     f"\t- Initial adaptation time: {initial_adaptation} s ({initial_adaptation_frames} frames)\n"
                     f"\n"
                     f"\t- N sequences: {n_sequences} ({nreps} repetitions --> {tot_seqs} sequences)\n"
                     f"\t- Total duration of the stimulus: {tot_stim_duration_s // 60}'{tot_stim_duration_s % 60}s ({tot_stim_duration_frames} frames)\n"
                     f"\t- Output folder: {output_folder}\n")
    print(report_string)
    print(f"Code to run: "
          f"GENERATE_NPY={GENERATE_NPY}, GENERATE_BIN={GENERATE_BIN}, "
          f"GENERATE_VEC={GENERATE_VEC}, VISUALIZE_STIMULUS_GIF={VISUALIZE_STIMULUS_GIF}")
    # ask user to confirm
    ok = input("\nContinue? (y/n): ")
    if ok.lower() != "y":
        print("Aborted.")
        return

    # ---------------------------------------------------------------------------------------------- #

    general_utils.make_dir(output_folder)
    general_utils.make_dir(files_folder)
    general_utils.write_json(parameters, param_path)
    if fig_traj is not None: general_utils.save_figure(fig_traj, f"Trajectories.jpg", output_folder)

    # >> STIMULUS GENERATION

    # - Store trajectories
    general_utils.save_dict(full_trajectories, trajectories_fp)

    if GENERATE_NPY:
        print("\n\nGENERATING NPY FILES")

        sid = 1
        ref_vec = {'SID': [], 'SYMBOL': [], 'SPEED (px/s)': [], 'SIZE (µm)': [], 'TRAJECTORY': [], 'TRJ_DURATION (frames)': []}
        for symbol in symbols:
            for speed, trajectories in full_trajectories.items():
                for symbol_size_um, symbol_size_px in zip(symbol_sizes_um, symbol_sizes_px):
                    print(f"\n\tProcessing symbol {symbol}, speed {speed} px/s, size {symbol_size_um} µm")
                    for trx_name, trx in tqdm(trajectories.items()):

                        # Generate and store stack of frames
                        npy_stack_frames = np.array([
                            symbol_stimuli_utils.generate_symbol_frame(symbol, SA_x_size_px, SA_y_size_px, symbol_size_px,
                                                                       symb_center, symbol_color, background_color, digit_type=dtype)
                            for symb_center in trx
                        ], dtype=dtype)

                        trj_duration_frames = npy_stack_frames.shape[0]
                        assert trj_duration_frames <= max_trj_duration_frames, f"Error: {trj_duration_frames} > {max_trj_duration_frames}"

                        npy_stack_frames_fn = f"{sid}_MS_seq.npy"
                        npy_stack_frames_fp = os.path.join(files_folder, npy_stack_frames_fn)
                        np.save(npy_stack_frames_fp, npy_stack_frames)

                        # Update reference vector
                        ref_vec['SID'].append(sid)
                        ref_vec['SYMBOL'].append(symbol)
                        ref_vec['SPEED (px/s)'].append(speed)
                        ref_vec['SIZE (µm)'].append(symbol_size_um)
                        ref_vec['TRAJECTORY'].append(trx_name)
                        ref_vec['TRJ_DURATION (frames)'].append(trj_duration_frames)

                        # Visualize sequence (store gif file)
                        if STORE_SINGLE_SEQUENCE_GIF:
                            gif_name = f"{sid}_{symbol}__Speed_{speed}_pxs__Size_{symbol_size_um}_um__{trx_name}.gif"
                            gif_fp = os.path.join(files_folder, gif_name)
                            gif.create_gif(npy_stack_frames, gif_fp, dt=dt * 1000, loop=1)

                        sid += 1

        # Check
        assert sid == n_sequences + 1, f"Error: {sid -1} != {n_sequences}"
        assert len(ref_vec['SID']) == n_sequences, f"Error: {len(ref_vec['SID'])} != {n_sequences}"

        # Save the reference table
        reference_table = pd.DataFrame(ref_vec)
        reference_table.to_csv(reference_table_fp, index=False)
        print(f" Reference table: {n_sequences} sequences x ({list(reference_table.columns)})")
        print(f" --> {reference_table_fp}")

    if GENERATE_BIN:
        print(f"\n\nGENERATING FRAME STACK (NPY and BIN)")
        from pycodes.modules.binfile import BinFile

        expected_dim = ((n_sequences * max_trj_duration_frames) + 1, SA_y_size_px, SA_x_size_px)

        # Initialize the frame stack and add the void frame in 0 position
        bin_frame_stack = [np.ones((SA_y_size_px, SA_x_size_px))*background_color]

        # Add all the sequences to the frame stack
        for sid in range(1, n_sequences + 1):
            npy_stack_frames = np.load(os.path.join(files_folder, f"{sid}_MS_seq.npy"))

            # Pad to zero the frames that are shorter than the longest one
            if npy_stack_frames.shape[0] < max_trj_duration_frames:
                npy_stack_frames = np.pad(npy_stack_frames, ((0, max_trj_duration_frames - npy_stack_frames.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=background_color)
            bin_frame_stack.extend(npy_stack_frames)
        bin_frame_stack = np.array(bin_frame_stack, dtype=dtype)

        assert bin_frame_stack.shape == expected_dim, f"Error: {bin_frame_stack.shape} != {expected_dim}"

        # Save the frame stack
        np.save(bin_frame_stack_fp, bin_frame_stack)
        print(f" Bin frame stack: {bin_frame_stack.shape}\n"
              f" --> {bin_frame_stack_fp}\n")

        bin_file = BinFile(file_bin_fp,
                           SA_y_size_px, SA_x_size_px,
                           nb_images=len(bin_frame_stack),
                           rig_id=RIG_ID,
                           mode='w')

        # write frames
        for i_frame in range(bin_frame_stack.shape[0]):
            bin_file.append_frame(bin_frame_stack[i_frame, :, :])

        # Close file
        bin_file.close()
        print(f" --> {file_bin_fp}")

    if GENERATE_VEC:
        print(f"\n\nGENERATING VEC FILE")
        tot_n_frames = tot_stim_duration_frames
        n_step_seq_binfile = max_trj_duration_frames
        reference_table = pd.read_csv(reference_table_fp)

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
            N_frames_sequence = reference_table.loc[reference_table['SID'] == sequence, 'TRJ_DURATION (frames)'].values[0]
            sequences_to_vec += N_frames_sequence * [sequence]

            start_on_bin = 1 + (sequence - 1) * n_step_seq_binfile
            end_on_bin = start_on_bin + N_frames_sequence
            frames_to_vec += list(range(start_on_bin, end_on_bin))

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
        show_up_to = int(tot_stim_duration_frames * 0.01)

        print(f"\n\nVISUALIZING STIMULUS GIF")
        print(f" --> Showing up to {show_up_to} frames ({show_up_to/stimulus_frequency} s)")

        frame_stack = np.load(bin_frame_stack_fp)
        vec = np.genfromtxt(vec_fp)
        frames_to_vec = vec[1:, 1]

        frames_to_vec = frames_to_vec[:show_up_to]

        complete_stack_frame = np.array([frame_stack[int(f), :, :] for f in frames_to_vec], dtype=dtype)

        assert complete_stack_frame.shape[0] == show_up_to, f"Error: {complete_stack_frame.shape[0]} != {show_up_to}"

        gif_fp = os.path.join(output_folder, f"{Stimulus_ID}.gif")
        gif.create_gif(complete_stack_frame, gif_fp, dt=dt * 1000, loop=1)
        print(f"Gif: {complete_stack_frame.shape} frames")
        print(f" --> {gif_fp}")

    return


if __name__ == '__main__':
    main()

