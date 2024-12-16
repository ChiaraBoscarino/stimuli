import numpy as np
import os
import matplotlib.pyplot as plt

from pycodes.modules.binfile import BinFile


GENERATE_BIN = True
CHECK_FROM_VEC = True
# INVERT_VEC = False


def main():
    stimulus_name = f"chirp_50Hz_32s_20reps_INVERTED"
    stimuli_folder = "D:\\STIMULI"  # "D:\\STIMULI" or "C:\\Users\\chiar\\Documents\\stimuli"
    output_folder = os.path.join(stimuli_folder, "Chirp")
    vec_fp = os.path.join(stimuli_folder, "Chirp", "chirp_50Hz_32s_20reps_NOT_INVERTED.vec")
    # vec_fp = os.path.join(stimuli_folder, "Chirp", "Euler_50Hz_20reps_1024x768pix.vec")

    if GENERATE_BIN:
        RIG_ID = 1
        x_size_px = 768
        y_size_px = 768
        color_range = [0, 255]
        N_sample_colorscale = 256  # (0, N-1)

        # generate colorscale
        colorscale = np.linspace(color_range[0], color_range[1], N_sample_colorscale)

        npy_stack_frames = np.array(colorscale[:, np.newaxis, np.newaxis] * np.ones((1, x_size_px, y_size_px)), dtype=np.uint8)
        tot_frames = npy_stack_frames.shape[0]
        binfile_fp = os.path.join(output_folder, f"{stimulus_name}.bin")
        bin_file = BinFile(binfile_fp,
                           frame_xsize=x_size_px,
                           frame_ysize=y_size_px,
                           nb_images=tot_frames,
                           rig_id=RIG_ID,
                           mode='w')
        for frame in npy_stack_frames:
            bin_file.append_frame(frame)
        bin_file.close()
        print(f"Generated bin file: ({npy_stack_frames.shape}) --> {binfile_fp}")

        if CHECK_FROM_VEC:
            # read vec
            vec_file = np.genfromtxt(vec_fp)
            frame_sequence = vec_file[1:, 1]
            min_value, max_value = np.min(frame_sequence), np.max(frame_sequence)

            if max_value > N_sample_colorscale-1:
                raise ValueError(f"Max value in vec file must be <= N-1  --> {max_value} >= N: {N_sample_colorscale}")

            # generate plot
            fig, axs = plt.subplots(2, 1, figsize=(30, 5))
            axs[0].plot(frame_sequence)
            axs[0].set_title(f"Frame sequence: {min_value} - {max_value}")
            axs[1].plot([colorscale[int(frame)] for frame in frame_sequence])
            axs[1].set_title(f"Color sequence: {color_range[0]} - {color_range[1]}")
            plt.show()

    # if INVERT_VEC:
    #     # Load vec file and invert it
    #     vec_file = np.genfromtxt(vec_fp)
    #     frame_sequence = vec_file[1:, 1]
    #     inverted_frame_sequence = np.max(frame_sequence) - frame_sequence
    #
    #     # generate plot
    #     fig, axs = plt.subplots(2, 1, figsize=(30, 5))
    #     axs[0].plot(frame_sequence)
    #     axs[0].set_title(f"Frame sequence: {np.min(frame_sequence)} - {np.max(frame_sequence)}")
    #     axs[1].plot(inverted_frame_sequence)
    #     axs[1].set_title(f"Color sequence: {np.min(inverted_frame_sequence)} - {np.max(inverted_frame_sequence)}")
    #     plt.show()

    return


if __name__ == '__main__':
    main()
