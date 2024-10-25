import numpy as np
import os
import matplotlib.pyplot as plt
import Moving_symbols_stimulus_parameters as prm
from pycodes.modules import general_utils, symbol_stimuli_utils

VISUALIZE_TRAJECTORIES = True
GENERATE_PARAMETER_FILE = True
GENERATE_NPY_FILE = True
GENERATE_BIN_FILE = True
GENERATE_VEC_FILE = True


def main():

    # VARIABLES
    root = 'C:\\Users\\chiar\\Documents\\stimuli'
    stim_type = 'MovingSymbols'
    output_folder = os.path.join(root, stim_type, prm.STIMULUS_VERSION_ID)
    # - parameters file
    param_source_fp = os.path.join(root, "pycodes", "Symbols", "Moving_symbols_stimulus_parameters.py")
    param_dest_fp = os.path.join(str(output_folder), f"{prm.STIMULUS_VERSION_ID}_parameters.py")
    # - vec file
    vec_fp = os.path.join(output_folder, f"{prm.STIMULUS_VERSION_ID}.vec")

    # CODE
    # - Output folder creation
    general_utils.make_dir(os.path.join(root, stim_type))
    general_utils.make_dir(output_folder)

    # - Trajectories computation
    symbol_center_trajectory = []
    for trajectory in prm.symbol_trajectories:
        trj = []
        for (x, y) in trajectory:
            # check trj point in [0, 1]
            if x < 0 or x > 1 or y < 0 or y > 1: raise ValueError("Trajectory coordinates must be between 0 and 1")
            # compute trj point in pixels on the SA
            trj_point_px = (int(x*prm.SA_x_size_px), int(y*prm.SA_y_size_px))
            # check trj point in SA for all symbol sizes
            for symbol_size_px in prm.symbol_sizes_px:
                symbol_stimuli_utils.symbol_location_check(trj_point_px, symbol_size_px, prm.SA_x_size_px, prm.SA_y_size_px)
            # append trj point to the trajectory
            trj.append(trj_point_px)
        # append the trajectory to the list of trajectories
        symbol_center_trajectory.append(trj)

    # - Show trajectories
    if VISUALIZE_TRAJECTORIES:
        ncols = min(5, len(symbol_center_trajectory))
        nrows = int(np.ceil(len(symbol_center_trajectory) / ncols))
        dim = 4
        fontsize = 12
        fig = plt.figure(figsize=(ncols*dim, nrows*dim))
        for i_t, trajectory in enumerate(symbol_center_trajectory):
            ax = fig.add_subplot(nrows, ncols, i_t+1)
            ax.set_xlim(0, prm.SA_x_size_px)
            ax.set_ylim(0, prm.SA_y_size_px)
            ax.invert_yaxis()  # invert y-axis to have (0, 0) in the top left corner
            x, y = zip(*trajectory)
            ax.plot(x, y, '-o', linewidth=1, markersize=17, markerfacecolor='white', color='k')
            for i_tp, trj_pt in enumerate(trajectory): ax.text(trj_pt[0], trj_pt[1], f"{i_tp}", fontsize=fontsize, ha='center', va='center')
            ax.set_title(f"Trajectory {i_t+1}", fontsize=fontsize)
        plt.show()

    # - User confirmation
    print(f"\n>> Generating Drifting Gratings stimulus (Version ID {prm.STIMULUS_VERSION_ID})\n"
          f"\t- Stimulus frequency: {prm.stimulus_frequency} Hz\n"
          f"\t- Frame size: {prm.SA_x_size_um}x{prm.SA_y_size_um} µm ({prm.SA_x_size_px}x{prm.SA_y_size_px} px)\n"
          
          f"\t- Symbols: {prm.symbols}\n"
          f"\t- Symbol sizes: {prm.symbol_sizes_um} µm ({prm.symbol_sizes_px} px)\n"
          
          f"\t- Symbol trajectories: {len(symbol_center_trajectory)} ({np.unique([len(x) for x in symbol_center_trajectory])} points)\n"
          f"\t- Symbol speeds: {np.array(prm.symbol_speeds)*prm.pixel_size} µm/s ({prm.symbol_speeds} px/s)\n"
          
          # f"\t- Total duration of the stimulus: {tot_stim_duration_s//60}'{tot_stim_duration_s%60}s ({tot_stim_duration_frames} frames)\n"
          
          f"\t- Output folder: {output_folder}\n")
    to_do = "\nWill run:\n"
    if GENERATE_PARAMETER_FILE:
        to_do += "\t- parameters file generation\n"
    if GENERATE_NPY_FILE:
        to_do += "\t- npy stack of frames generation\n"
    if GENERATE_BIN_FILE:
        to_do += "\t- bin file generation\n"
    if GENERATE_VEC_FILE:
        to_do += "\t- vec file generation\n"
    print(to_do)
    ok_to_go = input("Do you want to proceed? (y/n): ")
    if ok_to_go.lower() != 'y':
        print("Exiting...")
        return
    print("\n\n")

    if GENERATE_PARAMETER_FILE:
        print("Generating parameter file")
        import shutil
        shutil.copy(param_source_fp, param_dest_fp)

    if GENERATE_NPY_FILE:
        print("Generating npy file")
        # ADD HERE!!

    if GENERATE_BIN_FILE:
        print("Generating bin file")
        # ADD HERE!!

    if GENERATE_VEC_FILE:
        print("Generating vec file")
        # ADD HERE!!
        # tot_frames = 1000
        # frames_to_vec = []
        # sequences_to_vec = []

        # vec table generation
        vec = np.empty((tot_frames + 1, 5))  # create empty array with an extra row for the header
        vec[0, :] = [0, tot_frames, 0, 0, 0]  # fill the header
        vec[1:, :] = 0  # fill the rest of the array with zeros
        vec[1:, 1] = frames_to_vec  # fill the frames column (exact frame nb to show)
        vec[1:, 4] = sequences_to_vec  # fill the sequences column (ongoing sequence nb)

        # vec table to .vec file
        with open(vec_fp, "w") as f:
            np.savetxt(f, vec, delimiter=',', fmt='%i %i %i %i %i')
        print(f" Saved file: {vec_fp}")

    return


if __name__ == '__main__':
    main()

