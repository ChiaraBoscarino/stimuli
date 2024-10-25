import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

from pycodes.modules.gif import create_gif
from pycodes.modules import general_utils

# Code parameters
show_DG_parameters = False

generate_npy_stack_frames = False
show_all_frames = False

generate_bin_file = False

generate_vec = False

generate_report_sequence = False


def propagation_vector(direction, L):
    """ Compute the propagation vector for a given direction and spatial frequency.
    Args:
        direction: direction in degrees
        L: spatial frequency in px/cycle

    Returns:
        kx, ky: the propagation vectors in the x and y directions
    """
    kx = 2 * np.pi * np.cos(direction / 180 * np.pi) / L
    ky = 2 * np.pi * np.sin(direction / 180 * np.pi) / L
    return kx, ky


def convert_cpd_to_ppc(cpd, um_per_deg, pixel_size):
    """ Convert cycles per degree of visual angle to cycles per pixel.
    """
    return 1/(cpd/um_per_deg*pixel_size)


def get_stack_name(direction, L, s):
    f = s / L  # Temporal frequency in Hz
    return f"DG_{int(direction)}deg_{int(L)}ppc_{int(s)}pxs_{int(f)}Hz"


def main():

    # --------------------------------------- SET PARAMETERS --------------------------------------------- #
    # DG VERSION
    Version = "v2_MEA1"

    # Setup
    RIG_ID = 1
    if RIG_ID == 1 or RIG_ID == 2:
        pixel_size = 3.5  # in micrometers per pixel
    elif RIG_ID == 3:
        pixel_size = 2.8  # in micrometers per pixel
    else:
        raise ValueError("RIG_ID not recognized")

    # Direction
    N_directions = 4  # Number of directions
    directions = np.arange(0, 360, 360 / N_directions)

    # NB: a cycle is a full period of the wave, i.e. 2*pi
    # since we are using sin, it is from the start of the black to the end of the white

    # Spatial frequency
    wavelenghts_cpd = [0.1, 0.045, 0.03, 0.02, 0.012]  # in cycles per degree (how many full cycles in 1 degree of visual angle)
    um_per_deg = 32  # in micrometers per degree (projection of 1 deg of visual angle on the mouse retina)
    wavelengths_ppc = [int(convert_cpd_to_ppc(x, um_per_deg, pixel_size)) for x in wavelenghts_cpd]  # in px/cycle (how many pixels per cycle)

    # Temporal frequency
    wave_speeds = [50, 100, 150]  # in px/s

    # Sequences
    N_seq_tot = N_directions * len(wavelengths_ppc) * len(wave_speeds)
    N_repetitions = 8

    # Duration of the stimulus
    Tot_dur = 2  # seconds
    stimulus_frequency = 50  # Hz
    dt = 1 / stimulus_frequency  # timestep of a frame in seconds
    # t = np.arange(0, Tot_dur, dt)  # time vector
    N_frames = Tot_dur * stimulus_frequency  # Number of frames
    tot_frames = N_frames * len(directions) * len(wavelengths_ppc) * len(wave_speeds)
    tot_stim_duration = Tot_dur * N_repetitions * len(directions) * len(wavelengths_ppc) * len(wave_speeds)

    # Image
    frame_x_size_px = 768
    frame_y_size_px = 768

    # Source/Output folders
    output_folder = "C:\\Users\\chiar\\Documents\\rgc_typing\\stimuli\\DriftingGratings"
    general_utils.make_dir(output_folder)
    output_folder = os.path.join(output_folder, f"DG_{Version}")
    files_folder = os.path.join(output_folder, "files")
    general_utils.make_dir(output_folder)
    general_utils.make_dir(files_folder)
    stimulus_name = f"DG_stack_frames_{Version}_{stimulus_frequency}Hz"
    binfile_fp = os.path.join(output_folder, f"{stimulus_name}.bin")
    ref_vec_fp = os.path.join(output_folder, f"ref_vec_{stimulus_name}.csv")
    vec_fp = os.path.join(output_folder, f"vec_{stimulus_name}.vec")
    report_fp = os.path.join(output_folder, f"report_sequence_{stimulus_name}.csv")

    # ------------------------------------------------------------------------------------------ #

    # --------------------------------------- CODE --------------------------------------------- #

    print(f"\n>> Generating Drifting Gratings stimulus (Version {Version})\n"
          f"\t- Directions: {directions}\n"
          f"\t- Spatial frequencies: {wavelengths_ppc} px/cycle\n"
          f"\t- Temporal frequencies: {wave_speeds} px/s\n"
          f"\t- Stimulus frequency: {stimulus_frequency} Hz\n"
          f"\t- Total number of sequences: {N_seq_tot}\n"
          f"\t- Duration of each DG sequence: {Tot_dur} s\n"
          f"\t- Number of frames per sequence: {N_frames}\n"
          f"\t- Total number of frames (for the bin file): {tot_frames} (N seq. * N frames per seq)\n"
          f"\t- Frame size: {frame_x_size_px}x{frame_y_size_px} px\n"
          f"\t- Total duration of the stimulus: {tot_stim_duration} s ({tot_stim_duration//60}'{tot_stim_duration%60}s) (exp: {tot_frames/stimulus_frequency*N_repetitions})\n"
          f"\t- Output folder: {output_folder}\n")
    to_do = "\nWill run:\n"
    if show_DG_parameters:
        to_do += "\t- show DG parameters\n"
    if generate_npy_stack_frames:
        to_do += "\t- npy stack of frames generation\n"
    if generate_bin_file:
        to_do += "\t- bin file generation\n"
    if generate_vec:
        to_do += "\t- vec file generation\n"
    if generate_report_sequence:
        to_do += "\t- report sequence generation\n"
    print(to_do)
    ok_to_go = input("Do you want to proceed? (y/n): ")
    if ok_to_go.lower() != 'y':
        print("Exiting...")
        return

    # SHOW SELECTED STIMULI PARAMETERS OVERVIEW
    if show_DG_parameters:

        n_rows = 1
        n_cols = 2
        sq_dim = 5
        fontsize = 11
        fontsize_labels = fontsize - 2
        fig = plt.figure(figsize=(n_cols * sq_dim, n_rows * sq_dim))

        # Directions
        ax = fig.add_subplot(121)
        sq_lim = 0.1
        for direction in directions:
            kx, ky = propagation_vector(direction, 100)
            ax.quiver(0, 0, kx, ky, angles='xy', scale_units='xy', scale=1, color='black')
            ax.text(kx/2, ky/2, f"{direction}°", fontsize=fontsize, color='black', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=1))
        ax.set_xlim(-sq_lim, sq_lim)
        ax.set_ylim(-sq_lim, sq_lim)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=fontsize_labels)
        ax.set_title("DG directions", fontsize=fontsize)
        ax.set_axis_off()

        # Spatial/Temporal frequencies
        off_set = (0, 7)
        ax = fig.add_subplot(122)
        for L in wavelengths_ppc:
            for s in wave_speeds:
                f = s / L  # frequency in Hz (1/s)
                ax.scatter(L, s, s=100)
                ax.text(L+off_set[0], s+off_set[1], f"{f:.2f} Hz", fontsize=fontsize_labels, color='black', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=1))
        ax.set_xlabel("Spatial frequency (px/cycle)", fontsize=fontsize_labels, labelpad=10)
        ax.set_ylabel("Wave speed (px/s)", fontsize=fontsize_labels, labelpad=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.2)
        ax.set_title("DG spatial/temporal frequencies", fontsize=fontsize)
        r_x = max(wavelengths_ppc)-min(wavelengths_ppc)
        r_y = max(wave_speeds)-min(wave_speeds)
        ax.set_xlim(min(wavelengths_ppc)-0.2*r_x, max(wavelengths_ppc)+0.2*r_x)
        ax.set_ylim(min(wave_speeds)-0.2*r_y, max(wave_speeds)+0.2*r_y)

        plt.show()

    if generate_npy_stack_frames:
        print("\n>> Generating npy stack of frames")

        sid = 0
        ref_vec = {'sid': [], 'direction': [], 'L': [], 's': [], 'f': []}
        for L in wavelengths_ppc:
        # for L in Ls[0:1]:  # Only subset for now
            for s in wave_speeds:
            # for freq in freqs[0:2]:  # Only subset for now

                T = L / s  # period of the wave in seconds ([px/cycle]/[px/s] = s/cycle)
                f = 1 / T  # frequency in Hz (1/s)
                w = 2 * np.pi * f  # temporal frequency in 1/seconds (rad/s)

                print(f"Processing L={L:.2f} px/cycle, speed={s:.2f} px/s, freq={f:.2f} Hz")
                for direction in directions:
                # for direction in directions[0:1]:  # Only subset for now

                    npy_stack_frames = []

                    # propagating vector
                    kx, ky = propagation_vector(direction, L)

                    x_vector = np.arange(frame_x_size_px)  # row vector
                    y_vector = np.arange(frame_y_size_px)[:, None]  # column vector

                    for i_frame in (np.arange(0, N_frames)):
                        t = i_frame * dt
                        frame_img = np.where(np.sin(kx * x_vector + ky * y_vector - w * t) >= 0, 0, 1).astype(np.uint8)
                        npy_stack_frames.append(frame_img)

                    npy_stack_frames = np.array(npy_stack_frames)
                    stack_name = get_stack_name(direction, L, s)
                    npy_stack_frames_fn = f"{sid}_DG_seq.npy"
                    npy_stack_frames_fp = os.path.join(files_folder, npy_stack_frames_fn)
                    np.save(npy_stack_frames_fp, npy_stack_frames)
                    print(f"\t\t- ({sid}) {stack_name}: {npy_stack_frames.shape} shape (expected: {(N_frames, frame_y_size_px, frame_x_size_px)}) --> {npy_stack_frames_fn}")
                    ref_vec['sid'].append(sid)
                    ref_vec['direction'].append(direction)
                    ref_vec['L'].append(L)
                    ref_vec['s'].append(s)
                    ref_vec['f'].append(f)

                    # show all frames in a gif  to store
                    if show_all_frames:
                        gif_name = f"{sid}_{stack_name}.gif"
                        gif_fp = os.path.join(files_folder, gif_name)
                        create_gif(npy_stack_frames, gif_fp, dt=dt*1000, loop=1)

                    sid += 1

        ref_vec = pd.DataFrame(ref_vec)
        ref_vec.to_csv(ref_vec_fp, index=False)
        print(f"Saved reference vector to {ref_vec_fp}")

        assert sid == N_seq_tot, f"Expected {N_seq_tot} sequences, but last sid {sid}"

    if generate_bin_file:
        print("\n>> Generating bin file")
        from pycodes.modules.binfile import BinFile

        bin_file = BinFile(binfile_fp,
                           frame_xsize=frame_x_size_px,
                           frame_ysize=frame_y_size_px,
                           nb_images=tot_frames,
                           rig_id=RIG_ID,
                           mode='w')

        for sid in range(N_seq_tot):
            npy_stack_frames_fn = f"{sid}_DG_seq.npy"
            npy_stack_frames_fp = os.path.join(files_folder, npy_stack_frames_fn)
            npy_stack_frames = np.load(npy_stack_frames_fp)
            for frame in npy_stack_frames:
                bin_file.append_frame(frame)
        bin_file.close()

    if generate_vec:
        print("\n>> Generating vec file")

        sequences = pd.read_csv(ref_vec_fp)['sid'].values
        sequences_with_repetitions = np.repeat(sequences, N_repetitions)
        randomized_sequences = np.random.permutation(sequences_with_repetitions)
        randomized_sequences = general_utils.fix_adjacent_duplicates(randomized_sequences)
        if np.any(np.diff(randomized_sequences) == 0):
            print("WARNING: there are still adjacent sid in the randomized sequences")

        sequences_to_vec = []
        frames_to_vec = []
        for sid in randomized_sequences:
            sequences_to_vec += [sid] * N_frames

            seq_start_idx_binfile = sid * N_frames
            seq_end_idx_binfile = (sid + 1) * N_frames
            frames_to_vec += list(range(seq_start_idx_binfile, seq_end_idx_binfile))

        n_frames_displayed = len(randomized_sequences) * N_frames
        vec = np.empty((n_frames_displayed + 1, 5))
        vec[0, :] = [0, n_frames_displayed, 0, 0, 0]
        vec[1:, :] = 0
        vec[1:, 1] = frames_to_vec
        vec[1:, 4] = sequences_to_vec
        with open(vec_fp, "w") as f:
            np.savetxt(f, vec, delimiter=',', fmt='%i %i %i %i %i')
        print(f"Generated vec file: ({np.array(vec).shape}) --> {vec_fp}")

    if generate_report_sequence:
        vec_table = np.genfromtxt(vec_fp)
        # Read vec file
        vec_sequence = vec_table[1:, :]
        vec_header = vec_table[0, :]  # all zeros except for vec_header[1] = tot num of frames (== vec_table.shape[0]-1)

        sequence_ids = vec_sequence[:, -1]
        sequence_ids_unique = [int(sequence_ids[i]) for i in range(0, len(sequence_ids), N_frames)]

        # Read ref_vec file
        ref_vec_table = pd.read_csv(ref_vec_fp)
        ref_vec_table = ref_vec_table.set_index('sid')

        report = {'sid': [], 'direction': [], 'L': [], 's': [], 'f': []}
        for x in sequence_ids_unique:
            ref_vec_table_row = ref_vec_table.loc[x]
            report['sid'].append(x)
            report['direction'].append(ref_vec_table_row['direction'])
            report['L'].append(ref_vec_table_row['L'])
            report['s'].append(ref_vec_table_row['s'])
            report['f'].append(ref_vec_table_row['f'])
        report = pd.DataFrame(report)
        report.to_csv(report_fp, index=False)

    return


if __name__ == "__main__":
    main()
