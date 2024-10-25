import os
import numpy as np

from pycodes.modules import gif

seq_to_show = [1, 2, 3, 4, 92, 93, 94, 95]  # seconds
stimulus_version = "MSF_checkerboard_V10_386208"
# stimulus_version = "MSF_checkerboard_V9"
stimulus_name = f"{stimulus_version}_4Hz"
root = "C:\\Users\\chiar\\Documents\\rgc_typing"

frequency = 4  # Hz
sequence_duration_frames = 60 + 90  # frames
sid_rep_seq_cc = 0
sid_rep_seq_msc = 91
max_sid = 181


def main():

    stimulus_folder = os.path.join(root, "stimuli", stimulus_version)
    files_folder = os.path.join(stimulus_folder, "files")

    vec_file = f"{stimulus_name}.vec"
    vec_fp = os.path.join(stimulus_folder, vec_file)
    vec_table = np.genfromtxt(vec_fp)

    # Read vec file
    vec_sequence = vec_table[1:, :]
    vec_header = vec_table[0, :]  # all zeros except for vec_header[1] = tot num of frames (== vec_table.shape[0]-1)

    frames_to_vec = vec_sequence[:, 1]  # frame id
    Sequence_to_vec = vec_sequence[:, -1]  # sequence id

    spot_sequence_ids_unique = [int(Sequence_to_vec[i]) for i in range(sequence_duration_frames-1, len(Sequence_to_vec), sequence_duration_frames)]

    frames_sequence_rep_cc = np.load(os.path.join(files_folder, f"cc_sequence_0.npy"))
    frames_sequence_rep_msc = np.load(os.path.join(files_folder, f"msc_sequence_0.npy"))

    stack_to_show = []
    for sid in seq_to_show:
        if sid_rep_seq_cc < sid < sid_rep_seq_msc:
            frames_sequence = np.load(os.path.join(files_folder, f"cc_sequence_{sid}.npy"))
            frames_sequence_rep = frames_sequence_rep_cc
        elif sid_rep_seq_msc < sid <= max_sid:
            sid = sid - sid_rep_seq_msc
            frames_sequence = np.load(os.path.join(files_folder, f"msc_sequence_{sid}.npy"))
            frames_sequence_rep = frames_sequence_rep_msc
        else:
            raise ValueError(f"Sequence id {sid} out of range")

        for x in frames_sequence_rep: stack_to_show.append(x)
        for x in frames_sequence: stack_to_show.append(x)
    stack_to_show = np.array(stack_to_show)
    gif_fp = os.path.join(stimulus_folder, f"{stimulus_name}_sequence_some_sequences.gif")
    gif.create_gif(stack_to_show, gif_fp, dt=25, loop=1)

    return


if __name__ == '__main__':
    main()
