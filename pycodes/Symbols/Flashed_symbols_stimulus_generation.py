import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pycodes.modules import general_utils, symbol_stimuli_utils

# ---------------------------------------------------------------------------------------------- #
# >> CODE MODULATORS
GENERATE_NPY = True
# ---------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------- #
# >> PARAMETERS
rootpath = "C:\\Users\\chiar\\Documents\\stimuli"
output_root_folder = rootpath

parameters = {
    "STIMULUS_VERSION_ID": "FlashedSymbols_67_300ms",

    "pixel_size": 3.5,  # in µm per pixel
    "stimulus_frequency": 15,  # Hz

    "SA_sq_dim": 768,  # pixels

    "symbols": ['F', 'T', 'I'],
    "symbol_sizes_um": [150, 300, 450],  # µm
    "symbol_locations": {"150um": [(0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)],
                         "300um": [(0.3, 0.3), (0.7, 0.7)],
                         "450um": [(0.5, 0.5)]},
    "onset_time": 0.067,  # seconds
    "offset_time": 0.300,  # seconds
    "symbol_color": 1,
    "background_color": 0,

    "n_repetitions": 20,
}

# ---------------------------------------------------------------------------------------------- #


def main():
    Stimulus_ID = parameters["STIMULUS_VERSION_ID"]
    pixel_size = parameters["pixel_size"]
    stimulus_frequency = parameters["stimulus_frequency"]
    SA_sq_dim = parameters["SA_sq_dim"]
    symbols = parameters["symbols"]
    symbol_sizes_um = parameters["symbol_sizes_um"]
    symbol_locations = parameters["symbol_locations"]
    onset_time = parameters["onset_time"]
    offset_time = parameters["offset_time"]
    symbol_color = parameters["symbol_color"]
    background_color = parameters["background_color"]
    nreps = parameters["n_repetitions"]

    dt = 1 / stimulus_frequency  # timestep of a frame in seconds
    onset_time_frames = int(onset_time * stimulus_frequency)  # frames
    offset_time_frames = int(offset_time * stimulus_frequency)  # frames
    single_sequence_duration = onset_time + offset_time  # seconds

    SA_x_size_px, SA_y_size_px = SA_sq_dim, SA_sq_dim  # pixels
    SA_x_size_um, SA_y_size_um = SA_x_size_px * pixel_size, SA_y_size_px * pixel_size  # micrometers
    parameters["SA_x_size_um"], parameters["SA_y_size_um"] = SA_x_size_um, SA_y_size_um

    symbol_locations_px = {k: [(x*SA_x_size_px, y*SA_y_size_px) for (x,y) in locs] for k, locs in symbol_locations.items()}
    symbol_locations_um = {k: [(x*pixel_size, y*pixel_size) for (x,y) in locs] for k, locs in symbol_locations_px.items()}
    symbol_sizes_px = np.array([int(x // pixel_size) for x in symbol_sizes_um])  # pixels
    parameters["symbol_sizes_px"] = symbol_sizes_px
    n_sequences_per_symbol = np.sum([len(locs) for _, locs in symbol_locations.items()])
    tot_seqs = len(symbols) * n_sequences_per_symbol * nreps
    tot_dur = tot_seqs * single_sequence_duration  # seconds

    output_folder = os.path.join(output_root_folder, Stimulus_ID)
    param_path = os.path.join(output_folder, f"{Stimulus_ID}_parameters.json")
    frame_stack_fp = os.path.join(output_folder, f"{Stimulus_ID}_frame_stack.npy")
    reference_table_fp = os.path.join(output_folder, f"{Stimulus_ID}_reference_table.csv")
    vec_fp = os.path.join(output_folder, f"{Stimulus_ID}_vec.vec")
    general_utils.write_json(parameters, param_path)
    general_utils.make_dir(output_folder)

    # ---------------------------------------------------------------------------------------------- #
    # >> CODE
    print(f"\n\nGENERATING STIMULUS: {Stimulus_ID}")
    print(f" - Pixel size: {pixel_size} µm/px")
    print(f" - Stimulation area: {SA_x_size_um}x{SA_y_size_um} µm ({SA_x_size_px}x{SA_y_size_px} px)")
    print(f" - Symbols: {symbols}")
    print(f" - Symbol sizes: {symbol_sizes_um} µm")
    print(f" - Symbol locations: {symbol_locations_um}")
    print(f" - Stimulus frequency: {stimulus_frequency} Hz (dt: {dt} s)")
    print(f" - Onset time: {onset_time} s, offset time: {offset_time} s ({onset_time_frames}, {offset_time_frames} frames)")
    print(f" - N sequences: {tot_seqs} ({nreps} repetitions)")
    print(f" - Tot duration: {tot_dur//60}' {tot_dur%60} s")
    print(f" - Symbol color: {symbol_color} (background: {background_color})")
    # ask user to confirm
    ok = input("Continue? (y/n): ")
    if ok.lower() != "y":
        print("Aborted.")
        return

    # - GENERATE NPY
    if GENERATE_NPY:
        print(f"\n\nGENERATING FRAME STACK (NPY)")

        # Initialize the frame stack
        flash_frame_stack = np.zeros((tot_seqs, SA_x_size_px, SA_y_size_px))
        # add the void frame in 0 position
        flash_frame_stack[0, :, :] = np.zeros((SA_x_size_px, SA_y_size_px))

        # Initialize the reference table
        reference_table = {'SID': [], 'SYMBOL': [], 'SIZE (µm)': [], 'LOCATION (µm)': []}

        sid = 1
        for symbol in symbols:
            for symbol_size_um, symbol_size_px in zip(symbol_sizes_um, symbol_sizes_px):
                symbol_loc_px = symbol_locations_px[f"{symbol_size_px}um"]
                symbol_loc_um = symbol_locations_um[f"{symbol_size_px}um"]
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

        # Save the frame stack
        np.save(frame_stack_fp, flash_frame_stack)
        print(f" Flash frame stack: {flash_frame_stack.shape}\n"
              f" --> {frame_stack_fp}\n")

        # Save the reference table
        reference_table = pd.DataFrame(reference_table)
        reference_table.to_csv(reference_table_fp, index=False)
        print(f" Reference table: {sid} sequences x ({reference_table.columns})")
        print(f" --> {reference_table_fp}")

    return


if __name__ == "__main__":
    main()
