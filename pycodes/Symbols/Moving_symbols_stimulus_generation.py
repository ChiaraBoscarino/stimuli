import numpy as np
import os
import Moving_symbols_stimulus_parameters as prm
from pycodes.modules import general_utils, symbol_stimuli_utils

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
    general_utils.make_dir(stim_type)
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

    # - User confirmation
    print(f"\n>> Generating Drifting Gratings stimulus (Version ID {prm.STIMULUS_VERSION_ID})\n"
          # ADD HERE!!
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

