import random
import math
import struct

from tqdm.auto import tqdm
import copy

from pycodes.modules.binfile import *
from pycodes.modules.general_utils import *


def generate_2d_white_noise_frame(dimensions=(768, 768), seed=0):
    np.random.seed(seed)
    white_noise_2d_frame = np.random.choice([0, 1], size=(dimensions[0], dimensions[1]))
    return white_noise_2d_frame


def generate_2d_white_noise(nb_frames=300, dimensions=(768, 768), seed=0):
    """ Creates a sequence of images of white noise

    :param nb_frames: the number of frames generated, first dimension of the output array
    :param dimensions: the x and y dimensions of the images generated, respectively second and third dimensions of the output array
    :param seed: the seed used for random generation
    :return: white_noise_2d. A 3d array of dimensions (nb_frames, dimensions[0], dimensions[1]). Values are equal to 0 or 255.
    """
    np.random.seed(seed)
    white_noise_2d = np.random.choice([0, 1], size=(nb_frames, dimensions[0], dimensions[1]))
    return white_noise_2d


def upscale_2D_frame(frame, new_dimension):
    """ Upscales a 2d frame.
        The new y dimension should be a multiple of the original sequence y dimension.

        Args:
        - frame (np.array): a 2d array
        - new_dimension (int): the new y dimension
    """
    ratio = int(new_dimension / frame.shape[0])
    assert type(ratio) is int, "The ratio of the original and new y dimension should be an integer"
    if ratio > 1:
        upscaled_frame = np.kron(frame, np.ones((ratio, ratio)))
        upscaled_frame = upscaled_frame.astype("uint8")
    else:
        upscaled_frame = frame
    return upscaled_frame


def upscale_3D_array(array, new_dimension):
    """ Upscales a 3D array along the two last axes.
        These two last dimensions should be equal, and the new dimension should be a multiple.
    """
    assert array.shape[1] == array.shape[2]
    ratio = int(new_dimension / array.shape[1])
    assert type(ratio) is int, "The ratio of the original and new y dimension should be an integer"
    upscaled_array = np.repeat(array, ratio, axis=1)
    upscaled_array = np.repeat(upscaled_array, ratio, axis=2)
    return upscaled_array


def tilt_frame(frame, xtilt=0, ytilt=0):
    out_frame = np.empty((frame.shape[0], frame.shape[1]))

    for i in range(frame.shape[0]):
        x_tilted = i + xtilt
        if x_tilted < 0:
            x_tilted = frame.shape[0] - x_tilted
        if x_tilted >= frame.shape[0]:
            x_tilted = x_tilted - frame.shape[0]

        for j in range(frame.shape[1]):
            y_tilted = j + ytilt
            if y_tilted < 0:
                y_tilted = frame.shape[1] - y_tilted
            if y_tilted >= frame.shape[1]:
                y_tilted = y_tilted - frame.shape[1]

            out_frame[i, j] = frame[x_tilted, y_tilted]

    return out_frame


def scale_0_to_255(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = math.floor(255 * array[i, j])

    array = array.astype(int)
    return array


def write_bin_file(bin_filename, nbTotalFrames):
    TotalX = 864
    TotalY = 864
    nbTotalFrames = nbTotalFrames
    nBit = 8
    with open("{}.bin".format(bin_filename), "wb") as fb:
        fb.write(struct.pack("H", TotalX))
        fb.write(struct.pack("H", TotalY))
        fb.write(struct.pack("H", nbTotalFrames))
        fb.write(struct.pack("H", nBit))


def randomize_stimulus(nb_sequences=40, nb_repetitions=30):
    """Creates a randomized sequence of repeated sequences.
    Sequences are identified by integers ranging from 1 to the number of sequences.

    """
    ordered_sequences = np.array([i for i in range(1, nb_sequences + 1)])
    repeated_sequences = np.tile(ordered_sequences, nb_repetitions)
    randomized_sequences = np.random.permutation(repeated_sequences)
    return randomized_sequences


"""
Creates a stimulus composed of the classic checkerboard (cc) and the Multi Scale Checkerboard


"""
gen_cc_npy = True
gen_msc_npy = True
gen_bin = True
gen_vec = True

# Parameters

# Global
stimulus_random_number = np.random.randint(0, 1000000)
msf_id = f"MSF_checkerboard_V10_MEA1_{stimulus_random_number}"
root = "C:\\Users\\chiar\\Documents\\rgc_typing"  # !! SET THIS for windows
# root = r'/home/idv-equipe-s8/Documents/GitHub/rgc_typing'  # !! SET THIS for linux
stimuli_dir = os.path.join(root, 'stimuli')
msf_dir = os.path.join(stimuli_dir, msf_id)
files_dir = str(os.path.join(msf_dir, "files"))
make_dir(msf_dir)
make_dir(files_dir)

n_pixels = 768  # pixels
pixel_size = 3.5  # µm
rig_id = 1

# Classic checkerboard parameters
cc_check_size_mum = 56  # µm
cc_n_non_rep_seq = 90  # sequences
cc_n_frames_in_rep_seq = 60  # images
cc_n_frames_in_non_rep_seq = 90  # images

# MSF checkerboard
msc_check_sizes_mum = [56, 112, 224, 448, 896, 1344]  # µm (list)
msc_n_non_rep_seq = 90  # sequences
msc_n_frames_in_rep_seq = 60  # images
msc_n_frames_in_non_rep_seq = 90  # images

# vec
n_blocks = 3
framerate = 4

# Hardcoded parameters: don't change without adapting the codes accordingly
cc_n_rep_seq = 1  # sequences
msc_n_rep_seq = 1  # sequences

# Compute params
cc_check_size = int(round(cc_check_size_mum / pixel_size))
cc_n_checks = int(n_pixels / cc_check_size)
cc_n_images = cc_n_rep_seq * cc_n_frames_in_rep_seq \
              + cc_n_non_rep_seq * cc_n_frames_in_non_rep_seq

msc_n_images = msc_n_rep_seq * msc_n_frames_in_rep_seq \
               + msc_n_non_rep_seq * msc_n_frames_in_non_rep_seq
msc_check_sizes = [int(round(i / pixel_size)) for i in msc_check_sizes_mum]  # Convert to check sizes in pixel
msc_n_checks = [n_pixels / i for i in msc_check_sizes]  # compute number of checks on one side for each check size
msc_smallest_tilt = msc_check_sizes[0]  # in pixels
# Check that the values are correct
for n_check in msc_n_checks:
    assert n_check.is_integer()
msc_n_repeated_images_by_check_size = int(msc_n_frames_in_rep_seq / len(msc_check_sizes))
msc_n_non_repeated_images_by_check_size = int(msc_n_frames_in_non_rep_seq / len(msc_check_sizes))

n_images_total = msc_n_images + cc_n_images

n_frames_displayed = cc_n_non_rep_seq * cc_n_frames_in_rep_seq + \
                     cc_n_non_rep_seq * cc_n_frames_in_non_rep_seq + \
                     msc_n_non_rep_seq * msc_n_frames_in_rep_seq + \
                     msc_n_non_rep_seq * msc_n_frames_in_non_rep_seq

cc_n_seq_by_block = int(cc_n_frames_in_non_rep_seq / n_blocks)
msc_n_seq_by_block = int(msc_n_frames_in_non_rep_seq / n_blocks)


# Generate and save cc images as .npy
if gen_cc_npy:

    # Repeated sequences
    for i_sequence in tqdm(range(0, cc_n_rep_seq), desc='Create cc repeated sequences'):
        sequence = generate_2d_white_noise(nb_frames=cc_n_frames_in_rep_seq,
                                           dimensions=(cc_n_checks, cc_n_checks),
                                           seed=i_sequence).astype("uint8")
        upsampled_sequence = np.zeros((sequence.shape[0], n_pixels, n_pixels))
        for i_frame in range(sequence.shape[0]):
            upsampled_sequence[i_frame, :, :] = upscale_2D_frame(sequence[i_frame], n_pixels)
        np.save(os.path.join(files_dir, f"cc_sequence_{i_sequence}"), upsampled_sequence)

    # Non repeated sequences
    for i_sequence in tqdm(range(cc_n_rep_seq, cc_n_non_rep_seq + cc_n_rep_seq), desc='Create cc non repeated sequences'):
        sequence = generate_2d_white_noise(nb_frames=cc_n_frames_in_non_rep_seq,
                                           dimensions=(cc_n_checks, cc_n_checks),
                                           seed=i_sequence).astype("uint8")
        upsampled_sequence = np.zeros((sequence.shape[0], n_pixels, n_pixels))
        for i_frame in range(sequence.shape[0]):
            upsampled_sequence[i_frame, :, :] = upscale_2D_frame(sequence[i_frame], n_pixels)
        np.save(os.path.join(files_dir, f"cc_sequence_{i_sequence}"), upsampled_sequence)


# Generate and save msc images as .npy
if gen_msc_npy:

    # Generate repeated sequence

    # Initialize the seeds that will be used for random generation and tilt (incremented after each random draft)
    seed_for_frames = 1000
    seed_for_tilts = 1000

    for i_sequence in tqdm(range(0, msc_n_rep_seq)):

        # Create containers
        sequence = np.zeros((msc_n_frames_in_rep_seq, n_pixels, n_pixels), dtype='uint8')
        sequence_pattern = np.zeros(msc_n_frames_in_rep_seq)
        sequence_tilts = np.zeros((msc_n_frames_in_rep_seq, 2))
        # Prepare a list with the size of the checks for each frame (shuffled)
        check_sizes_nbs = []
        for i in msc_check_sizes:
            check_sizes_nbs += msc_n_repeated_images_by_check_size * [i]
        random.seed(0)
        random.shuffle(check_sizes_nbs)

        for frame_nb in range(0, msc_n_frames_in_rep_seq):
            check_size = check_sizes_nbs[0]  # Get the first check size value
            if len(check_sizes_nbs) > 1:
                check_sizes_nbs = check_sizes_nbs[1:]  # Remove this value from the list
            sequence_pattern[frame_nb] = check_size  # Save the check size to the
            dimension = int(n_pixels / check_size)
            # Create the white noise frame
            frame = generate_2d_white_noise_frame(dimensions=(dimension, dimension), seed=seed_for_frames)
            seed_for_frames += 1  # Increment the seed so that the next frame is different
            upsampled_frame = upscale_2D_frame(frame,
                                               n_pixels)  # If the checks are not the smallest size, upscale the frame so that it matches the intermediate root dimension

            # Randomly draw x and y axis tilts
            np.random.seed(seed_for_tilts)
            xtilt = np.random.randint(0, int(n_pixels / msc_smallest_tilt)) * msc_smallest_tilt  # in pixels
            seed_for_tilts += 1
            np.random.seed(seed_for_tilts)
            ytilt = np.random.randint(0, int(n_pixels / msc_smallest_tilt)) * msc_smallest_tilt  # in pixels
            seed_for_tilts += 1
            sequence_tilts[frame_nb, 0] = xtilt
            sequence_tilts[frame_nb, 1] = ytilt
            upsampled_tilted_frame = tilt_frame(upsampled_frame, xtilt=xtilt, ytilt=ytilt)

            sequence[frame_nb, :, :] = upsampled_tilted_frame

        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}"), sequence)
        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}_check_sizes"), sequence_pattern)
        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}_tilts"), sequence_tilts)

    # Generate non repeated sequence

    # Initialize the seeds that will be used for random generation and tilt (incremented after each random draft)
    seed_for_frames = 2000
    seed_for_tilts = 2000

    for i_sequence in tqdm(range(msc_n_rep_seq, msc_n_non_rep_seq + msc_n_rep_seq)):

        # Create containers
        sequence = np.zeros((msc_n_frames_in_non_rep_seq, n_pixels, n_pixels), dtype='uint8')
        sequence_pattern = np.zeros(msc_n_frames_in_non_rep_seq)
        sequence_tilts = np.zeros((msc_n_frames_in_non_rep_seq, 2))
        sequence_seeds = np.zeros(msc_n_frames_in_non_rep_seq)

        # Prepare a list with the size of the checks for each frame (shuffled)
        check_sizes_nbs = []
        for i in msc_check_sizes:
            check_sizes_nbs += msc_n_non_repeated_images_by_check_size * [i]
        random.seed(i_sequence)
        random.shuffle(check_sizes_nbs)

        for frame_nb in range(0, msc_n_frames_in_non_rep_seq):
            check_size = check_sizes_nbs[0]  # Get the first check size value
            if len(check_sizes_nbs) > 1:
                check_sizes_nbs = check_sizes_nbs[1:]  # Remove this value from the list
            sequence_pattern[frame_nb] = check_size  # Save the check size to the
            dimension = int(n_pixels / check_size)  # n checks on one side
            # Create the white noise frame
            sequence_seeds[frame_nb] = seed_for_frames
            frame = generate_2d_white_noise_frame(dimensions=(dimension, dimension), seed=seed_for_frames)
            seed_for_frames += 1  # Increment the seed so that the next frame is different
            upsampled_frame = upscale_2D_frame(frame,
                                               n_pixels)  # If the checks are not the smallest size, upscale the frame so that it matches the intermediate root dimension

            # Randomly draw x and y axis tilts
            np.random.seed(seed_for_tilts)
            xtilt = np.random.randint(0, int(n_pixels / msc_smallest_tilt)) * msc_smallest_tilt  # in pixels
            seed_for_tilts += 1
            np.random.seed(seed_for_tilts)
            ytilt = np.random.randint(0, int(n_pixels / msc_smallest_tilt)) * msc_smallest_tilt  # in pixels
            seed_for_tilts += 1
            sequence_tilts[frame_nb, 0] = xtilt
            sequence_tilts[frame_nb, 1] = ytilt
            upsampled_tilted_frame = tilt_frame(upsampled_frame, xtilt=xtilt, ytilt=ytilt)

            sequence[frame_nb, :, :] = upsampled_tilted_frame

        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}"), sequence)
        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}_check_sizes"), sequence_pattern)
        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}_tilts"), sequence_tilts)
        np.save(os.path.join(files_dir, f"msc_sequence_{i_sequence}_seeds"), sequence_seeds)


# Write all the frames in a bin file
if gen_bin:

    bin_file = BinFile(os.path.join(msf_dir, f"{msf_id}_{framerate}Hz.bin"),
                       n_pixels,
                       n_pixels,
                       nb_images=n_images_total,
                       rig_id=rig_id,
                       mode='w')

    # Write cc sequences
    for i_sequence in tqdm(range(cc_n_rep_seq + cc_n_non_rep_seq),
                           desc="Writing cc sequences to bin"):
        sequence = np.load(os.path.join(files_dir, f"cc_sequence_{i_sequence}.npy"))
        for i_frame in range(sequence.shape[0]):
            bin_file.append_frame(sequence[i_frame, :, :])

    # Write msc sequences
    for i_sequence in tqdm(range(msc_n_rep_seq + msc_n_non_rep_seq),
                           desc="Writing msc sequences to bin"):
        sequence = np.load(os.path.join(files_dir, f"msc_sequence_{i_sequence}.npy"))
        for i_frame in range(sequence.shape[0]):
            bin_file.append_frame(sequence[i_frame, :, :])

    bin_file.close()


# Generate the vec file
if gen_vec:

    cc_repeated_id = 0
    msc_repeated_id = cc_n_non_rep_seq + msc_n_rep_seq
    cc_non_repeated_id = [i for i in range(cc_n_rep_seq, cc_n_non_rep_seq + cc_n_rep_seq)]
    msc_non_repeated_id = [i for i in range(cc_n_non_rep_seq + msc_n_rep_seq + 1,
                                            cc_n_non_rep_seq + msc_n_rep_seq + msc_n_non_rep_seq + 1)]

    cc_nr_tmp = copy.deepcopy(cc_non_repeated_id)
    msc_nr_tmp = copy.deepcopy(msc_non_repeated_id)

    random.seed(0)
    random.shuffle(cc_non_repeated_id)
    random.shuffle(msc_non_repeated_id)

    sequences = []
    for block in range(n_blocks):
        for i in range(cc_n_seq_by_block):
            sequences += [cc_repeated_id]
            sequences += [cc_nr_tmp[0]]
            cc_nr_tmp = cc_nr_tmp[1:]

        for i in range(msc_n_seq_by_block):
            sequences += [msc_repeated_id]
            sequences += [msc_nr_tmp[0]]
            msc_nr_tmp = msc_nr_tmp[1:]

    sequences_to_vec = []
    frames_to_vec = []

    for sequence in sequences:

        # Update sequence to vec
        if sequence == cc_repeated_id:
            sequences_to_vec += cc_n_frames_in_rep_seq * [sequence]
        elif sequence == msc_repeated_id:
            sequences_to_vec += msc_n_frames_in_rep_seq * [sequence]
        elif sequence in cc_non_repeated_id:
            sequences_to_vec += cc_n_frames_in_non_rep_seq * [sequence]
        elif sequence in msc_non_repeated_id:
            sequences_to_vec += msc_n_frames_in_non_rep_seq * [sequence]
        else:
            raise ValueError

        # Update the frames to vec
        if sequence == cc_repeated_id:
            first_frame = 0
            frames_to_vec += [i for i in range(first_frame, first_frame + cc_n_frames_in_rep_seq)]

        elif sequence in cc_non_repeated_id:
            first_frame = cc_n_frames_in_rep_seq + (sequence - 1) * cc_n_frames_in_non_rep_seq
            frames_to_vec += [i for i in range(first_frame, first_frame + cc_n_frames_in_non_rep_seq)]

        elif sequence == msc_repeated_id:
            first_frame = cc_n_frames_in_rep_seq + cc_n_non_rep_seq * cc_n_frames_in_non_rep_seq
            frames_to_vec += [i for i in range(first_frame, first_frame + msc_n_frames_in_rep_seq)]

        else:
            first_frame = cc_n_frames_in_rep_seq + cc_n_non_rep_seq * cc_n_frames_in_non_rep_seq + msc_n_frames_in_rep_seq + (
                    sequence - cc_n_non_rep_seq - 2) * msc_n_frames_in_non_rep_seq
            frames_to_vec += [i for i in range(first_frame, first_frame + msc_n_frames_in_non_rep_seq)]

        # print(sequence, len(frames_to_vec))

    # Create the vec array
    vec = np.empty((n_frames_displayed + 1, 5))

    # fill the header
    vec[0, :] = [0, n_frames_displayed, 0, 0, 0]

    # fill the vec array with the frame numbers
    vec[1:, :] = 0
    vec[1:, 1] = frames_to_vec
    vec[1:, 4] = sequences_to_vec

    # write the vec array in a .vec file
    with open(f"{msf_dir}/{msf_id}_{framerate}Hz.vec", "w") as f:
        np.savetxt(f, vec, delimiter=',', fmt='%i %i %i %i %i')

    print(f"The stimulus is composed of {n_frames_displayed} frames and lasts {n_frames_displayed / framerate / 60}min if displayed at {framerate}Hz.")
