import numpy as np
import os
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------------------------------------------- #
#   CHECKERBOARD STIMULUS
# ------------------------------------------------------------------------------------------------------------------- #


def check_msc_info(seq_id, source_dir="files"):
    fp_sizes = os.path.join(source_dir, f"msc_sequence_{seq_id}_check_sizes.npy")
    fp_tilts = os.path.join(source_dir, f"msc_sequence_{seq_id}_tilts.npy")
    sizes = np.load(fp_sizes)
    tilts = np.load(fp_tilts)

    if seq_id != 0:
        fp_seeds = os.path.join(source_dir, f"msc_sequence_{seq_id}_seeds.npy")
        seeds = np.load(fp_seeds)
    else:
        seeds = None

    # # REPORT
    # print(f"Sequence {seq_id}\n"
    #       f"\tSizes: {sizes}\n"
    #       f"\tTilts: {tilts}\n"
    #       f"\tSeeds: {seeds}\n")

    return sizes, tilts, seeds


def get_msc_sizes(source_dir="files"):
    all_sizes = []
    for seq_id in range(1, 90):
        sizes, _, _ = check_msc_info(seq_id, source_dir)
        all_sizes = np.concatenate((all_sizes, sizes))

    sizes, counts = np.unique(all_sizes, return_counts=True)

    return sizes, counts


def coord_pixel2unit(x_px, y_px, unit_per_px):
    """ Given the coordinates of a point in a reference frame in pixels,
        return the coordinates of the center of the center of the pixel on a unit-based reference frame.
        NB: both reference frames have the top-left corner is still the origin (0,0)
        and the axis are still oriented left- and down-wise
    """
    # multiply the coordinates in pixels per the number of units per pixel
    x_u = x_px * unit_per_px + (unit_per_px / 2)
    y_u = y_px * unit_per_px + (unit_per_px / 2)

    return x_u, y_u


def coord_check2unit(x_cks, y_cks, X_checks, Y_checks, X_unit, Y_unit):
    """ Given the coordinates of a check (x_cks, y_cks) on a checkerboard reference frame,
        (where the top-left corner is the (0,0) check, and check counts increase left- and down-wise)
        return the coordinates of the center of that check on a unit-based reference frame,
        (where the top-left corner is still the origin (0,0) and the axis are still oriented left- and down-wise).

        Args:
        - (x_cks, y_cks) : coordinates of the check on the checkerboard reference frame
        - (X_checks, Y_checks) : nb of checks per size of the checkerboard
        - (X_unit, Y_unit) : size dimension of the checkerboard in units

        Returns:
        - (x_u, y_u) : coordinates of the center of the check in the unit-based reference frame
        - (unit_per_check_x, unit_per_check_y) : check dimension in units
    """
    # Get the units per check on each side
    unit_per_check_x = X_unit / X_checks
    unit_per_check_y = Y_unit / Y_checks

    # multiply the coordinates in checks per the number of units per check
    # and add half the check dimension to compensate for the coordinates in checks
    # location at the center of the check
    x_u = (x_cks * unit_per_check_x) + (unit_per_check_x / 2)
    y_u = (y_cks * unit_per_check_y) + (unit_per_check_y / 2)

    return x_u, y_u, unit_per_check_x, unit_per_check_y


def rescale_checkerboard(cc, n_pixel_per_check_old, n_pixel_per_check_new):
    """Given a checkerboard (frames,n,m), the number of pixels per check in the old checkerboard and
        the number of pixels per check in the new checkerboard, returns the checkerboard rescaled
        to the new number of pixels per check.
    """
    if n_pixel_per_check_new == n_pixel_per_check_old:
        return cc

    if len(cc.shape) == 2:
        cc = cc[np.newaxis, :, :]

    n_frames, n_rows, n_cols = cc.shape
    n_rows_rescaled = int(n_rows * n_pixel_per_check_new / n_pixel_per_check_old)
    n_cols_rescaled = int(n_cols * n_pixel_per_check_new / n_pixel_per_check_old)
    cc_rescaled = np.zeros((n_frames, n_rows_rescaled, n_cols_rescaled))

    for f in np.arange(n_frames):
        for i in np.arange(n_rows_rescaled):
            for j in np.arange(n_cols_rescaled):
                cc_rescaled[f, i, j] = cc[f, int(i * n_pixel_per_check_old / n_pixel_per_check_new), int(
                    j * n_pixel_per_check_old / n_pixel_per_check_new)]

    return cc_rescaled


def get_checkerboard_sequence(filepath, rescale=False, n_pixel_per_check_old=None, n_pixel_per_check_new=None):
    """ Load from the npy files containing the classic checkerboards frames (filepath)
        the checkerboard sequence with the given id,
        eventually rescaling the result to a new number of pixels per check.

        Args:
        - filepath (str): path to the npy file containing the checkerboard sequence
        - rescale (bool): if True, rescale the checkerboard to a new number of pixels per check
        - n_pixel_per_check_old (int): number of pixels per check in the old checkerboard
        - n_pixel_per_check_new (int): number of pixels per check in the new checkerboard

        Returns:
        - checkerboard_sequence (np.array): 3D matrix (frames, rows, columns) containing the checkerboard sequence
    """
    checkerboard_sequence = np.load(filepath)
    if rescale:
        if n_pixel_per_check_new is None or n_pixel_per_check_old is None:
            raise ValueError("n_pixel_per_check_new and n_pixel_per_check_old must be specified to rescale")
        checkerboard_sequence = rescale_checkerboard(checkerboard_sequence, n_pixel_per_check_old,
                                                     n_pixel_per_check_new)
    return checkerboard_sequence


def image_projection(image, mea):
    """ Given an image and the mea type, return the image projected according to the mea type."""
    if mea == 2:
        image = np.rot90(image)
        image = np.flipud(image)

    elif mea == 3:
        image = np.fliplr(image)
    return image


def checkerboard_from_binary(nb_frames, nb_checks_x, nb_checks_y, checkerboard_file, binary_source_path, mea):
    """ Create a checkerboard stimulus from a binary file."""
    binary_source_file = open(binary_source_path, mode='rb')
    checkerboard = np.zeros((nb_frames, nb_checks_x, nb_checks_y), dtype='uint8')

    for frame in tqdm(range(nb_frames)):

        image = np.zeros((nb_checks_x, nb_checks_y), dtype=float)

        for row in range(nb_checks_x):
            for col in range(nb_checks_y):
                bit_nb = (nb_checks_x * nb_checks_y * frame) + (nb_checks_x * row) + col
                binary_source_file.seek(bit_nb // 8)
                byte = int.from_bytes(binary_source_file.read(1), byteorder='big')
                bit = (byte & (1 << (bit_nb % 8))) >> (bit_nb % 8)
                if bit == 0:
                    image[row, col] = 0.0
                elif bit == 1:
                    image[row, col] = 1.0
                else:
                    message = "Unexpected bit value: {}".format(bit)
                    raise ValueError(message)

        checkerboard[frame, :, :] = image_projection(image, mea)
    np.save(checkerboard_file, checkerboard)
    print(f'Checkerboard stimulus created and saved at : {checkerboard_file}')
    return checkerboard

# ------------------------------------------------------------------------------------------------------------------- #
#   MULTISIZE SPOT STIMULUS
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------ RETRIEVE FROM VEC FILE ------------------------- #

def get_frame_info(frames_reference, frame_id):
    """ Retrieve the information about a frame from the frames_reference file.
        The frames_reference file contains the information about the frames of the checkerboard sequence.
        Each row of the file contains the following information about a frame:
        - size (in checks)
        - x coordinate of the center of the frame (in checks)
        - y coordinate of the center of the frame (in checks)
        - x coordinate of the center of the frame (in unit)
        - y coordinate of the center of the frame (in unit)
        - x coordinate of the center of the frame (in unit, aligned to the center of the area)
        - y coordinate of the center of the frame (in unit, aligned to the center of the area)

        Args:
        - frames_reference (np.array): 2D matrix (num_frames, 7) containing the information about the frames
            Note: use the following codes to load the frames_reference file:
                reference_path = "stimulus_radial_error_0.2_frames_reference.vec"
                frames_reference = np.genfromtxt(reference_path)
        - frame_id (int): id of the frame

        Returns:
        - size (int): size of the frame (in checks)
        - x_original (int): x coordinate of the center of the frame (in checks)
        - y_original (int): y coordinate of the center of the frame (in checks)
        - x_actual (float): x coordinate of the center of the frame (in unit)
        - y_actual (float): y coordinate of the center of the frame (in unit)
        - x_aligned (float): x coordinate of the center of the frame (in unit, aligned to the center of the area)
        - y_aligned (float): y coordinate of the center of the frame (in unit, aligned to the center of the area)
    """
    size = frames_reference[frame_id, 0]
    x_original = frames_reference[frame_id, 1]
    y_original = frames_reference[frame_id, 2]
    x_actual = frames_reference[frame_id, 3]
    y_actual = frames_reference[frame_id, 4]
    x_aligned = frames_reference[frame_id, 5]
    y_aligned = frames_reference[frame_id, 6]
    return size, x_original, y_original, x_actual, y_actual, x_aligned, y_aligned


def spot_sequence_reconstruction(disk_stimulus, fid, duration_onset, duration_offset):
    """ Given the frame id, reconstruct the sequence actually shown to the cells.
        NB: if a frame is not in the disk stimulus (npy array),
        then a black sequence (as fid=0) is reconstructed.
    """
    if not 0 <= fid < disk_stimulus.shape[0]: fid = 0
    onset = np.tile(disk_stimulus[fid, :, :], (duration_onset, 1, 1))
    offset = np.tile(disk_stimulus[0, :, :], (duration_offset, 1, 1))
    spot_sequence = np.concatenate((onset, offset), axis=0)
    return spot_sequence


def get_frame_repetitions(frames_sequence, sequence, spot_sequence_duration, mode="SERIES", verbose=False):
    """ Given the frames_sequence and the spot_sequence for the spot stimulus,
    return the dictionary of the frames and the list of frames."""
    dict = {}

    # build the dictionary including the number of the
    if mode == "SERIES":
        for s, frame_id in enumerate(frames_sequence):
            if frame_id not in dict.keys():
                dict[frame_id] = []
            dict[frame_id] += [s]

    elif mode == "ONSETS":
        for i in range(0, len(sequence), spot_sequence_duration):
            frame_id = int(sequence[i])
            if frame_id not in dict.keys():
                dict[frame_id] = []
            dict[frame_id] += [i]

    else:
        raise ValueError("mode must be 'series' or 'onsets'")

    list_of_frames = list(dict.keys())
    list_of_frames.sort()

    if verbose:
        print(f"\n{mode}\n"
              f"\tNum. of frames: {len(dict)}\n"
              f"\tNum. of repetitions: {np.unique([len(dict[frame_id]) for frame_id in dict.keys()])}\n"
              f"\tFrames: from {min(list_of_frames)} to {max(list_of_frames)} "
              f"(step {np.unique(np.diff(list_of_frames))})\n")

    return dict, list_of_frames


def process_spot_stimulus_sta(spot_stimulus_npy_filename, frames_reference, cut_lower_edge_px, cut_upper_edge_px, spot_sta_selected_sizes, verbose=False):
    """ Given the spot stimulus npy file, return the processed spot stimulus for spot sta computation.
        The spot stimulus is processed by:
        - cutting the stimulus to get only the stimulated area
        - selecting the frames to consider for the sta computation (using only
            spots of the smallest sizes --> up to k-th size)
    """

    # LOAD DATA
    disk_stimulus = np.load(spot_stimulus_npy_filename)

    # CUTTING THE STIMULUS
    cut_stimulus = disk_stimulus[:, cut_lower_edge_px: cut_upper_edge_px, cut_lower_edge_px: cut_upper_edge_px]

    # SPOT FRAMES SELECTION
    selected_fids = []
    for fid in range(disk_stimulus.shape[0]):
        size, _, _, _, _, _, _ = get_frame_info(frames_reference, fid)
        if min(spot_sta_selected_sizes) <= size <= max(spot_sta_selected_sizes):
            selected_fids += [fid]
    size_filtered_stimulus = cut_stimulus[selected_fids, :, :]

    if verbose:
        print(f"Spot stimulus processed: "
              f"{disk_stimulus.shape[1]} px --> ({cut_lower_edge_px}, {cut_upper_edge_px}) in px --> {cut_stimulus.shape[1]}\n"
              f"\toriginal shape: {disk_stimulus.shape}\n"
              f"\tafter cut shape: {cut_stimulus.shape}\n"
              f"\tafter size selection shape: {size_filtered_stimulus.shape} (selected sizes: {spot_sta_selected_sizes})\n")

    return size_filtered_stimulus
