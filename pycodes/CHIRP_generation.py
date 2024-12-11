import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from pycodes.modules.binfile import BinFile
from pycodes.modules.general_utils import make_dir

CHECK_FROM_VEC = True
# vec_fp = "C:\\Users\\chiar\\Documents\\stimuli\\Chirp\\chirp_MEA1_50Hz_32s_20reps.vec"
vec_fp = "C:\\Users\\chiar\\Documents\\stimuli\\Chirp\\Euler_50Hz_20reps_1024x768pix.vec"


def main():
    stimulus_name = f"chirp_MEA1_50Hz_32s_20reps"
    stimuli_folder = "C:\\Users\\chiar\\Documents\\stimuli"
    output_folder = os.path.join(stimuli_folder, "Chirp")

    RIG_ID = 1
    x_size_px = 768
    y_size_px = 768
    color_range = [0, 1]
    N_sample_colorscale = 256  # (0, N-1)

    # generate colorscale
    colorscale = np.linspace(color_range[0], color_range[1], N_sample_colorscale)

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

    npy_stack_frames = np.array(colorscale[:, np.newaxis, np.newaxis] * np.ones((1, x_size_px, y_size_px)), dtype='uint8')
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

    return


if __name__ == '__main__':
    main()
