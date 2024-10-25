import os
import ast
import json
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
#     IMAGE FILTERING
# ----------------------------------------------------------------------------------------------------------------------
import cv2


def filter_image_sequence(images_sequence, kernel, pad_type="zero"):
    """
    Filter each frame of the sequence using the specified kernel.

    Parameters:
        images_sequence (numpy.ndarray): Input sequence of frames with shape (N, M, M).
        kernel (numpy.ndarray): Kernel for filtering with shape (X, X).

    Returns:
        filtered_sequence (numpy.ndarray): Filtered sequence of frames.
    """
    # Ensure the kernel is 2D
    if len(kernel.shape) != 2:
        raise ValueError("Kernel must be a 2D array.")
    # Ensure the sequence is 3D
    if len(images_sequence.shape) != 3:
        raise ValueError("Image sequence must be a 3D array.")

    if pad_type == "zero":
        border_type = cv2.BORDER_CONSTANT
    elif pad_type == "replicate":
        border_type = cv2.BORDER_REPLICATE
    else:
        raise ValueError("Pad type not recognized.")

    filtered_sequence = np.zeros_like(images_sequence)

    # Iterate over frames and filter each one
    for i in range(images_sequence.shape[0]):
        # Apply 2D convolution using cv2.filter2D
        filtered_frame = cv2.filter2D(images_sequence[i, :, :], -1, kernel, borderType=border_type)

        # Store the result in the output sequence
        filtered_sequence[i, :, :] = filtered_frame

    return filtered_sequence


# ----------------------------------------------------------------------------------------------------------------------
#     GENERAL
# ----------------------------------------------------------------------------------------------------------------------

def align_coordinates(x, y, x_center, y_center):
    """Given the coordinates of a point (x, y) in a reference frame centered in the top-left corner of the area,
        returns the coordinates of the point in the reference frame centered in (x_center, y_center).

        Args:
        - x (float): x coordinate of the point in the reference frame centered in the top-left corner of the area
        - y (float): y coordinate of the point in the reference frame centered in the top-left corner of the area
        - x_center (float): x coordinate of the center of the area in the reference frame centered in the top-left corner of the area
        - y_center (float): y coordinate of the center of the area in the reference frame centered in the top-left corner of the area

        Returns:
        - x_aligned (float): x coordinate of the point in the reference frame centered in (x_center, y_center)
        - y_aligned (float): y coordinate of the point in the reference frame centered in (x_center, y_center)
    """
    x_aligned = x - x_center
    y_aligned = -(y - y_center)
    return x_aligned, y_aligned


def euclidean_distance(x, y):
    """Calculates the Euclidean distance between two n-dimensional vectors x and y."""
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))


def get_start_end(s, n):
    """ Get the n start and n end element of an array, with 4 nan in the middle."""
    start = s[:int(n / 2)]
    mid = np.array([np.nan] * 4)
    end = s[-int(n / 2):]
    return np.concatenate((start, mid, end))


def array_to_vec_file(array, header, filename):
    ext = '.vec'
    if not (filename[-4:] == ext):
        filename += ext

    # write the vec array in a .vec file
    with open(filename, "w") as f:
        np.savetxt(f, array, delimiter=',', fmt='%i ' * array.shape[1], header=header)


def ask_recording_name(recording_names):
    """ Ask the user which recording he wants to analyze and return the recording name as a string.
    """
    l = "Select the recording name:\n"
    for num, rec in enumerate(recording_names):
        l += f"\n\t{num} --> {rec}"
    recording_number = int(input(f'{l}\n\nNumber : '))

    rec_name = recording_names[recording_number]
    print('Selected recording: {}\n'.format(rec_name))

    return rec_name, recording_number


def fix_adjacent_duplicates(v):
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            swap_idx = i
            # Find the next non-duplicate to swap
            for j in range(i + 1, len(v)):
                if v[j] != v[i]:
                    swap_idx = j
                    break
            # Swap the elements
            v[i], v[swap_idx] = v[swap_idx], v[i]
    return v

# ----------------------------------------------------------------------------------------------------------------------
#     MULTIDIMENSIONAL LIST/ARRAYS
# ----------------------------------------------------------------------------------------------------------------------


def flatten_list(listoflists):
    """Flatten a list of lists"""
    flat_list = []
    for l in listoflists:
        flat_list.extend(l)
    return flat_list


def get_non_nan(matrix):
    """From a matrix with nans,
        return a vector with only the non-nan values of the matrix"""
    return matrix[~np.isnan(matrix)]


def count_nan(matrix):
    """ Count the number of nan values in a matrix"""
    return np.isnan(matrix).sum()


def get_vector_resolution(v, precision=8):
    """ Get the resolution of a vector (i.e. the distance between two consecutive values).
        The precision can be specified (default 1e-8)"""
    v = np.array(v)
    r = np.unique([round(x, precision) for x in v[1:] - v[:-1]])
    if len(r) > 1:
        raise ValueError(f"Vector resolution not uniform: {r}")
    return r[0]


def shift_in_pixel(shift_units, resolution_units):
    """ Compute the shift in terms of vector positions: given r the resolution of the vector,
        if between i and i+1 there are r units,
        then to perform  shift of x units the shift must be of x/r positions,
        however positions must be integers, so the result is rounded to the closest integer.

        Args:
        - shift_units: shift in units
        - resolution_units: resolution of the vector

        Returns:
        - shift_pixels: shift in pixels
    """
    shift_pixels = round(shift_units / resolution_units)
    return shift_pixels


def check_shape_uniformity(d, verbose=False):
    """ Check that all values from a dictionary of matrices have the same shape."""
    shapes = [m.shape for m in d.values()]
    if len(set(shapes)) > 1:
        raise ValueError(f"Matrix shapes not uniform: {shapes}")
    if verbose:
        print(f"Uniformity check OK: all values have shape {shapes[0]}")
    return shapes[0]

# ----------------------------------------------------------------------------------------------------------------------
#     SHIFT A MATRIX
# ----------------------------------------------------------------------------------------------------------------------


def m_shift(m, x, y, fillvalue=np.nan):
    """ Shift a matrix M by x columns and y rows.
        Specify the fillvalue for the introduced positions (default nan)."""
    if x >= 0:
        sm = shift(m, abs(x), how="right", fv=fillvalue)
    else:
        sm = shift(m, abs(x), how="left", fv=fillvalue)

    if y >= 0:
        sm = shift(sm, abs(y), how="down", fv=fillvalue)
    else:
        sm = shift(sm, abs(y), how="up", fv=fillvalue)

    return sm


def shift(matrix, n, how, fv):
    """ Shift a matrix M by n positions in the specified direction.
        Specify the fillvalue for the introduced positions."""
    if n == 0: return matrix
    if how == "left":
        m = np.roll(matrix, -n, axis=1)
        m[:, -n:] = fv
    elif how == "right":
        m = np.roll(matrix, n, axis=1)
        m[:, 0:n] = fv
    elif how == "up":
        m = np.roll(matrix, -n, axis=0)
        m[-n:, :] = fv
    elif how == "down":
        m = np.roll(matrix, n, axis=0)
        m[0:n, :] = fv
    else:
        raise ValueError(f"Unknown shift {how}")
    return m


# ----------------------------------------------------------------------------------------------------------------------
#     TO AND FROM FILE
# ----------------------------------------------------------------------------------------------------------------------

def save_figure(fig, filename, figuredir, remove_blank_space=False):
    """Save a figure in the specified directory"""
    filepath = os.path.join(figuredir, filename)
    if remove_blank_space:
        fig.savefig(filepath, bbox_inches='tight')
    else:
        fig.savefig(filepath)
    plt.close(fig)


def load_df(csv_path, index_col='', name='', verbose=False):
    """Load a .csv file as a pandas dataframe"""
    df = pd.read_csv(csv_path)
    if index_col != '': df.set_index(index_col, inplace=True)
    if verbose: print(f"\n\nLoaded {name}:\t\t{df.shape}\n")
    return df


def save_dict(dictionary, filename, outdir=None):
    """Save a dictionary (with pickle)"""
    save_with_pickle(filename, dictionary, outdir)


def save_var(var, filename, outdir=None):
    """Save a variable (with pickle)"""
    save_with_pickle(filename, var, outdir)


def load_var(pickle_path):
    """Load a variable (with pickle)"""
    return load_with_pickle(pickle_path)


def load_dict(pickle_path):
    return load_with_pickle(pickle_path)


def save_with_pickle(filename, var, outdir=None):
    """Save a variable with pickle"""
    ext = '.pkl'
    if filename[-len(ext):] != ext:
        filename += ext
    if outdir is not None:
        path = os.path.join(outdir, filename)
    else:
        path = filename
    with open(path, 'wb') as file:
        pickle.dump(var, file)
    print(f"\nSaved: {path}")


def load_with_pickle(pickle_path):
    """Load a variable with pickle"""
    ext = '.pkl'
    if pickle_path[-len(ext):] != ext:
        pickle_path += ext
    with open(pickle_path, 'rb') as file:
        return pickle.load(file)


def load_vector(filename, delimiter=','):
    """Load a .csv file containing a vector and return it as np.array"""
    # Loading .csv files containing a vector and returning it as np.array
    axis = np.genfromtxt(filename, delimiter=delimiter)
    return axis


def load_json(path):
    """Load a .json file and return the corresponding dictionary"""
    # Loading JSON files and returning the corresponding dictionary
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def generate_dictionary_from_csv(dir_name, extension='.csv', delimiter=','):
    """ Build a dictionary from a set of files.

        Args:
        - dir_name: directory containing the files
        - extension: file extension (default: .csv)
        - delimiter: delimiter used in the files (default: ',')

        Returns:
        - dictionary: {filename: np.array generated from the file text (using np.genfromtxt, with the specified delimiter))}
    """
    # Initialize the dictionary
    d = {}
    # Loop over the files in the directory
    for file in os.listdir(dir_name):
        # Check if the file has the correct extension (else ignore it)
        if file.endswith(extension):
            # Read the file
            filename = os.path.join(dir_name, file)
            m = np.genfromtxt(filename, delimiter=delimiter)
            # Add the content to the dictionary
            d[file[:-len(extension)]] = m
    return d


# ----------------------------------------------------------------------------------------------------------------------
#     DICTIONARIES
# ----------------------------------------------------------------------------------------------------------------------

def get_keys(d, level):
    """Given a nested dictionary:
                d = { K1 : { K2 : ... } } or d = { K1 : { K2 : { K3 : ... } } }
        Return the keys at a specific level.
    """
    if level == 1:
        return list(d.keys())

    elif level == 2:
        k1 = list(d.keys())[0]
        return list(d[k1].keys())

    elif level == 3:
        k1 = list(d.keys())[0]
        k2 = list(d[k1].keys())[0]
        return list(d[k1][k2].keys())
    else:
        raise ValueError(f"ERROR: incorrect level {level} (try: 1, 2, 3)")


def get_unique_shape_in_dict_fixed_key_level1(d, key):
    """ Given a nested dictionary having for many elements the same dictionary:

        d = { A : { key: [...], ... },
              B : { key: [...], ... }, ...}

        return the unique shape of the list of elements d[x][key] for all x in d.keys()
    """
    # compute the shapes of the elements d[x][key] for all x in d.keys()
    unique_shapes = {np.shape(d[x][key]) for x in d.keys()}

    # check there is at least one shape
    if len(unique_shapes) == 0:
        raise ValueError(f"ERROR in getting unique shape! Empty dictionary.")
    # check whether non-homogeneous shapes are found
    if len(unique_shapes) > 1:
        raise ValueError(f"ERROR in getting unique shape! Multiple shapes found: {unique_shapes}")
    # return the unique shape
    if len(unique_shapes) == 1:
        return unique_shapes.pop()
    else:
        raise ValueError(f"ERROR in getting unique shape! Unexpected behavior.")


def count_elements_in_dict(d, level):
    """ Given a nested dictionary:
            d = { K1 : { K2 : ... } } or d = { K1 : { K2 : { K3 : ... } } }
        Find the number of elements for each element in the dictionary.

        Args:
            - d: dictionary
            - level: level of the dictionary (1 for number elements in the dict (num. of keys K1),
                                                2 for number of K1 having n K2,
                                                3 for number of K2 having n K3)
        Returns:
            {n Kx: count} meaning that there are 'count' elements with 'n_elements_per_element' elements
    """
    if level == 1:
        return len(list(d.keys()))

    elif level == 2:
        n_elements_per_element, count = np.unique([len(d[k1].keys()) for k1 in d.keys()], return_counts=True)
        return dict(zip(n_elements_per_element, count))

    elif level == 3:
        nb_elem = []
        for k1 in d.keys():
            for k2 in d[k1].keys():
                nb_elem.append(len(d[k1][k2].keys()))
        n_elements_per_element, count = np.unique(nb_elem, return_counts=True)
        return dict(zip(n_elements_per_element, count))
    else:
        raise ValueError(f"ERROR in count_elements_in_dict: incorrect level {level} (try: 1, 2, 3)")


def change_dict_key(d, old_key, new_key):
    """Change the key of an item within a dictionary"""
    if old_key in d.keys():
        d[new_key] = d.pop(old_key)
    else:
        raise ValueError(f"Key {old_key} not found in dictionary")


def n_items(d, n):
    """Return a dictionary with the first n items of the input dictionary"""
    return {k: d[k] for k in list(d.keys())[:n]}


# ORDER DICTIONARY
def sort_dict(d, by="values", mode="+"):
    """ Return a sorted dictionary

        Specify:
        - by: "keys" or "values"
        - mode: "+" or "-" (ascending or descending)
    """
    if by == "values":
        if mode == "+":  # ascending
            return dict(sorted(d.items(), key=lambda item: item[1]))
        elif mode == "-":  # descending
            return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        else:
            print("mode not found")
            return d

    elif by == "keys":
        if mode == "+":  # ascending
            return dict(sorted(d.items()))
        elif mode == "-":  # descending
            return dict(sorted(d.items(), reverse=True))
        else:
            print("mode not found")
            return d
    else:
        print("ordering criteria not found")
        return d


def print_dict(d, n=None):
    """ Print a dictionary.
        Specify the number of items to print (default: all)"""
    if n is None:
        n = len(d)
    for i, k in enumerate(d.keys()):
        if i >= n:
            break
        print(f"{k}: {d[k]}")


# ----------------------------------------------------------------------------------------------------------------------
#     RETRIEVE THE VAR FROM A VAR-LIKE STRINGS
# ----------------------------------------------------------------------------------------------------------------------


def string_to_dict(input_string):
    """Retrieve the dictionary from a dict-like string"""
    try:
        # Using ast.literal_eval to safely evaluate the string as a Python literal or container display
        dictionary = ast.literal_eval(input_string)
        if isinstance(dictionary, dict):
            return dictionary
        else:
            raise ValueError("Invalid input: Not a dictionary.")
    except (SyntaxError, ValueError) as e:
        print(f"Error: {e}")
        return None


def string_to_matrix(input_string):
    """Retrieve the matrix from a matrix-like string"""
    try:
        # Using ast.literal_eval to safely evaluate the string as a Python literal or container display
        var = ast.literal_eval(input_string)
        if isinstance(var, (list, tuple, np.ndarray)):
            try:
                np.array(var)
            except:
                print(f"Error: Impossible np.array conversion.")
                return None
            return var
        else:
            raise ValueError("Invalid input: Not a matrix.")
    except (SyntaxError, ValueError) as e:
        print(f"Error: {e}")
        return None


# ----------------------------------------------------------------------------------------------------------------------
#     DIRECTORY HANDLING
# ----------------------------------------------------------------------------------------------------------------------


def make_dir(new_dir):
    """Create a directory if it does not exist"""
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
        print("\tDirectory created: ", new_dir)
    else:
        print("\tDirectory already exists: ", new_dir)


# ----------------------------------------------------------------------------------------------------------------------
#     ASK USER INPUT
# ----------------------------------------------------------------------------------------------------------------------

def yn_input_mandatory_answer(question):
    """Ask a yes/no question to the user.
        The user must answer with 'y' or 'n',
        any other input is not accepted and the system will ask again."""
    text = question + " (y/n)"
    answer = input(text)
    while answer != 'y' and answer != 'n':
        answer = input(text)
    answer = True if answer == 'y' else False
    return answer


def choose_file_from_dir(src_folder, keyword=''):
    """ Ask the user to choose a file from a directory.
        Specify a keyword to filter the files in the directory (default: all files)."""
    print(f"\nAvailable files:")
    filelist = [x for x in os.listdir(src_folder) if keyword in x]
    for i, f in enumerate(filelist):
        print(f"{i}. {f}")

    x = int(input("\nChoose a file number: "))
    filename = filelist[x]
    if filename in os.listdir(src_folder):
        return filename
    else:
        raise ValueError(f"File {filename} not found")


# ----------------------------------------------------------------------------------------------------------------------
#     DATAFRAMES
# ----------------------------------------------------------------------------------------------------------------------


def unified_index_set(df_filepaths, index_col):
    """ Generate a unified set of indices from a set of dataframes

        Args:
        - df_filepath_set: collection of filepaths to the dataframes
        - index_col: name of the column containing the indices

        Returns:
        - index_set: set of indices common to all the dataframes
    """
    index_set = None
    for i, df_filepath in enumerate(df_filepaths):
        # load each df in the provided set
        df = load_df(df_filepath, index_col=index_col, verbose=False)
        # initialize the set of indices with the indices of the first df
        if i == 0:
            index_set = list(df.index)
        # for all other dfs in the set, intersect the set of indices with the indices of the current df
        else:
            index_set = [idx for idx in index_set if idx in list(df.index)]

    return index_set


# ----------------------------------------------------------------------------------------------------------------------
#     MATH and STATISTICS
# ----------------------------------------------------------------------------------------------------------------------

def correlate_PersonPM(cell1, cell2, max_shift=25):
    """ (From standard pipeline https://github.com/retinal-information-processing-lab/Standard_analysis_pipeline/blob/dev-guilhem/utils.py)
        Compute the Pearson correlation between two cells, with a maximum shift of max_shift time_bins.
        The correlation is computed for each shift in [-max_shift, max_shift] and the result is returned as an array.

        Args:
        - cell1, cell2: two cells to correlate
        - max_shift: maximum shift in time_bins

        Returns:
        - correlation: array of correlations for each shift
    """
    assert max_shift < max(len(cell1),len(cell2))
    center = np.corrcoef(cell1,cell2)[0,1]
    right = []
    left = []
    for t in range(1,max_shift+1):
        right.append(np.corrcoef(cell1[t:],cell2[:-t])[0,1])
        left.append(np.corrcoef(cell1[:-max_shift+t-1],cell2[max_shift-t+1:])[0,1])
    return np.asarray(left + [center] + right)

