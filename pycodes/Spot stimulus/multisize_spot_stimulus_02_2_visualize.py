import os
import numpy as np
import matplotlib.pyplot as plt

from pycodes.modules import general_utils, gif

n_seq_to_show = 10  # spots
stimulus_version = "MSSpots_V7_MEA1_30error_12reps_4Hz"
stimulus_name = "MSSpots_V7_MEA1_30error_12reps_4Hz"
stim_dir = '/home/idv-equipe-s8/Documents/GitHub/hiOsight/stimuli/MultisizeSpots'

sequence_duration_s = 2  # seconds
frequency = 4  # Hz
black_frame_fid = 0


def main():
    sequence_duration_frames = sequence_duration_s * frequency

    frame_stack_file = f"{stimulus_name}_frame_stack.npy"
    vec_file = f"{stimulus_name}.vec"
    spot_reference_filename = f"{stimulus_name}_spot_reference.vec"

    stimulus_folder = os.path.join(stim_dir, stimulus_version)
    vec_fp = os.path.join(stimulus_folder, vec_file)
    spot_reference_fp = os.path.join(stimulus_folder, spot_reference_filename)
    frames_stack_fp = os.path.join(stimulus_folder, frame_stack_file)

    vec_table = np.genfromtxt(vec_fp)
    frames_reference = np.genfromtxt(spot_reference_fp)
    frames_stack = np.load(frames_stack_fp)

    # Read vec file
    vec_sequence = vec_table[1:, :]
    vec_header = vec_table[0, :]  # all zeros except for vec_header[1] = tot num of frames (== vec_table.shape[0]-1)

    spot_frame_ids = vec_sequence[:, 1]  # first the id of the spot shown in the sequence and then the id of the black frame
    spot_sequence_ids = vec_sequence[:, -1]  # during the whole spot sequence onset + offset the number of the frame shown during the onset

    spot_sequence_ids_unique = [int(spot_sequence_ids[i]) for i in range(0, len(spot_sequence_ids), sequence_duration_frames)]

    stack_to_show = []
    for i in range(n_seq_to_show):
        fid = spot_sequence_ids_unique[i]
        stack_to_show.append(frames_stack[fid])
        stack_to_show.append(frames_stack[black_frame_fid])
    stack_to_show = np.array(stack_to_show)
    gif_fp = os.path.join(stimulus_folder, f"{stimulus_name}_sequence_first{n_seq_to_show}spots.gif")
    gif.create_gif(stack_to_show, gif_fp, dt=1000, loop=1)

    trajectory = frames_stack[black_frame_fid]
    for i in range(n_seq_to_show):
        v = i+1
        trajectory += frames_stack[spot_sequence_ids_unique[i]]*v
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cax = ax.imshow(trajectory, cmap='viridis')
    cbar = fig.colorbar(cax, ax=ax)
    plt.axis('off')
    general_utils.save_figure(fig, f"{stimulus_name}_sequence_first{n_seq_to_show}spots_trajectory.jpg", stimulus_folder)

    return


if __name__ == '__main__':
    main()
