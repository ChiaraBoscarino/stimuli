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


def rescale_sequence(cc, n_pixel_per_check_old, n_pixel_per_check_new):
    """ Given a sequence (frames,n,m), the number of pixels per check in the old sequence and
        the number of pixels per check in the new sequence, returns the sequence rescaled
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
        checkerboard_sequence = rescale_sequence(checkerboard_sequence, n_pixel_per_check_old, n_pixel_per_check_new)
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
