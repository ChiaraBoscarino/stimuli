import numpy as np
import os

from pycodes.modules.binfile import BinFile
from pycodes.modules.general_utils import make_dir
from pycodes.modules.symbol_stimuli_utils import generate_symbol_frame


def main():
    Test_stimulus_name = f"F_test_stimulus_MEA1"
    RIG_ID = 1
    x_size_px = 768
    y_size_px = 768
    symbol_size_px = 500
    symbol_color = 1
    background_color = 0
    symbol_center = (380, 380)  # (x, y) in px assuming y-axis pointing down and x-axis pointing right
    onset_duration = 5  # in seconds
    stimulus_freq = 40  # in Hz
    tot_frames = onset_duration * stimulus_freq
    stimuli_folder = "C:\\Users\\chiar\\Documents\\rgc_typing\\stimuli"
    output_folder = os.path.join(stimuli_folder, "Test_stimuli")
    make_dir(output_folder)

    print(f"Generating {Test_stimulus_name}:\n"
          f"Frame size: {x_size_px}x{y_size_px} px\n"
          f"Symbol size: {symbol_size_px} px\n"
          f"Symbol center: {symbol_center}\n"
          f"Symbol color: {symbol_color}\n"
          f"Background color: {background_color}\n"
          f"Duration: {onset_duration} s\n"
          f"Frequency: {stimulus_freq} Hz\n"
          f"Total frames: {tot_frames}\n")
    F_frame = generate_symbol_frame('F', x_size_px, y_size_px, symbol_size_px, symbol_center, symbol_color, background_color)
    npy_stack_frames = np.array([F_frame for _ in range(tot_frames)], dtype='uint8')
    binfile_fp = os.path.join(output_folder, f"{Test_stimulus_name}.bin")
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

    vec_fp = os.path.join(output_folder, f"{Test_stimulus_name}.vec")
    frames_to_vec = np.arange(0, tot_frames)
    sequences_to_vec = np.zeros(tot_frames, dtype=int)
    n_frames_displayed = tot_frames
    vec = np.empty((n_frames_displayed + 1, 5))
    vec[0, :] = [0, n_frames_displayed, 0, 0, 0]
    vec[1:, :] = 0
    vec[1:, 1] = frames_to_vec
    vec[1:, 4] = sequences_to_vec
    with open(vec_fp, "w") as f:
        np.savetxt(f, vec, delimiter=',', fmt='%i %i %i %i %i')
    print(f"Generated vec file: ({np.array(vec).shape}) --> {vec_fp}")
    return


if __name__ == '__main__':
    main()
