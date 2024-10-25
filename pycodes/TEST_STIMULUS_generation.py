import matplotlib.pyplot as plt
import numpy as np
import os

from pycodes.modules.binfile import BinFile
from pycodes.modules.general_utils import make_dir


def generate_symbol_frame(symbol, x_size_px, y_size_px, symbol_size_px, symbol_center, symbol_color, background_color):
    """ Generate a frame with a symbol with the specified size and location.
        The symbol is colored with the symbol_color and the frame is filled with the background color.

        Args:
            symbol:
            x_size_px, y_size_px: frame dimensions in pixels
            symbol_size_px: symbol dimension (square edge) in pixels
            symbol_center: symbol location in pixel coordinates [(x, y) assuming y-axis pointing down and x-axis pointing right]
            symbol_color: color value to fill the symbol with
            background_color: color value to fill the frame background with

        Returns:
            frame: numpy array (y_size_px, x_size_px) with the generated frame

    """
    frame = np.full((y_size_px, x_size_px), background_color, dtype=np.uint8)

    # Get top-left corner coordinates of the symbol (in pixels)
    symbol_center_x = symbol_center[0]
    symbol_center_y = symbol_center[1]
    symbol_top_left_x = symbol_center_x - symbol_size_px // 2
    symbol_top_left_y = symbol_center_y - symbol_size_px // 2
    if symbol_top_left_x < 0 or symbol_top_left_y < 0:
        raise ValueError("Symbol is out of frame boundaries")
    square_dim_row_start = symbol_top_left_y
    square_dim_row_end = symbol_top_left_y + symbol_size_px
    square_dim_col_start = symbol_top_left_x
    square_dim_col_end = symbol_top_left_x + symbol_size_px

    if symbol == 'F':
        bar_width = int(symbol_size_px * 0.2)
        # horizontal bar
        frame[square_dim_row_start:square_dim_row_start+bar_width, square_dim_col_start:square_dim_col_end] = symbol_color
        # vertical bar
        frame[square_dim_row_start: square_dim_row_end, square_dim_col_start:square_dim_col_start+bar_width] = symbol_color
        # small horizontal central bar
        frame[square_dim_row_start+bar_width*2:square_dim_row_start+bar_width*3, square_dim_col_start:square_dim_col_start+symbol_size_px//2] = symbol_color

    # show_frame
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    cax = ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
    ax.scatter(symbol_center_x, symbol_center_y, s=50, c='red', marker='o')
    ax.scatter(symbol_top_left_x, symbol_top_left_y, s=50, c='blue', marker='o')
    ax.add_patch(plt.Rectangle((square_dim_col_start, square_dim_row_start), symbol_size_px, symbol_size_px, edgecolor='gray', facecolor='none'))
    fig.colorbar(cax)
    ax.grid(True, which='both', linestyle='-', linewidth=0.2)
    plt.show()
    return frame


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