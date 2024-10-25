import matplotlib.pyplot as plt
import numpy as np
from pycodes.modules import multisize_spot_stimulus_utils


TITLE_PAD = 20
LABELS_FONT_SIZE = 12


def get_non_nan(matrix):
    """From a matrix with nans,
        return a vector with only the non-nan values of the matrix"""
    return matrix[~np.isnan(matrix)]


def get_range(list_of_matrices):
    """ Get (min, max) over all given matrices"""
    gmax = np.max(np.array([np.max(get_non_nan(d)) for d in list_of_matrices]))
    gmin = np.min(np.array([np.min(get_non_nan(d)) for d in list_of_matrices]))
    return gmin, gmax


def get_dynrange(list_of_matrices):
    """ Get symmetric dynamic range over all given matrices
        (the absolute value representing the symmetric range
        containing all values in provided matrices, namely the max absolute value)"""
    gmin, gmax = get_range(list_of_matrices)
    return max(abs(gmin), abs(gmax))


def color_ax_borders(ax, color, linewidth=5):
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(linewidth)


def adjust_plot(ax, x_axis_range, y_axis_range, xticks=None, yticks=None, x_label=None, y_label=None, title=''):
    """ Adjust the plot settings by:
        - setting the labels font size
        - setting axis limits

        Args:
            - ax: the axis to be adjusted
            - x_axis_range, y_axis_range: the limits of the axis
            - title: the title of the plot (OPTIONAL, if not provided '' is added)
    """
    # AXIS LABELS
    if x_label is not None:
        ax.set_xlabel(x_label)
        ax.xaxis.label.set_size(LABELS_FONT_SIZE)
    if y_label is not None:
        ax.set_ylabel(y_label)
        ax.yaxis.label.set_size(LABELS_FONT_SIZE)

    # if xticks is None:
    #     xticks = generate_numerical_ticks(x_axis_range, 5)
    #     xticks = [round(xtick, 2) for xtick in xticks]
    # if yticks is None:
    #     yticks = generate_numerical_ticks(y_axis_range, 5)
    # ax.set_xticklabels(xticks, fontsize=LABELS_FONT_SIZE*0.8)
    # ax.set_yticklabels(yticks, fontsize=LABELS_FONT_SIZE*0.8)

    # AXIS RANGE
    if ax.get_xlim() != x_axis_range: ax.set_xlim(x_axis_range)
    if ax.get_ylim() != y_axis_range: ax.set_ylim(y_axis_range)

    # REMOVE THE ROUNDING BOX (LEFT TOP)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # TITLE
    ax.set_title(title, pad=TITLE_PAD)

    return ax


def plot_frame_sequence(frame_sequence, cmap="gray", crange=None):
    """ Given a sequence of  images, plot all of them in sequence.
        If crange is specified, it is used to set the limits of the common colorbar,
        otherwise a common colorbar is defined using the max values in the images.
    """
    print("Frame sequence shape: ", frame_sequence.shape)

    nframes = frame_sequence.shape[0]
    ncols = 8
    nrows = int(np.ceil(nframes / ncols))
    imdim = 4
    fig = plt.figure(figsize=(ncols * imdim, nrows * imdim))

    if crange is None:
        crange = get_dynrange(frame_sequence)
    if cmap == "gray":
        cmaprange = [0, crange]
    else:
        cmaprange = [-crange, crange]

    for i in range(nframes):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(frame_sequence[i, :, :], cmap=cmap, vmin=cmaprange[0], vmax=cmaprange[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"f= {i}")
    plt.show()
    return


def add_spot_frame_to_figure(fid,
                             disk_stimulus, frames_reference,
                             size_stimulus, size_mea,
                             ax, grid=False):
    """ Add spot frame image to figure, with stimulation area centered.
        Also, the mea, the center of the spot and the center of the stimulation area are shown.
    """
    x_axis_range = (-size_stimulus / 2, size_stimulus / 2)
    y_axis_range = (-size_stimulus / 2, size_stimulus / 2)
    plot_extent = x_axis_range + y_axis_range
    scatter_size = (plt.rcParams['lines.markersize'] ** 2) / 4
    frame_to_plot = disk_stimulus[fid, :, :] if 0 < fid < disk_stimulus.shape[0] else disk_stimulus[0, :, :]
    size, _, _, _, _, x_aligned, y_aligned = multisize_spot_stimulus_utils.get_frame_info(frames_reference, fid)
    if grid: ax.grid()
    ax.imshow(frame_to_plot, cmap='gray', extent=plot_extent)
    ax.add_patch(plt.Rectangle((-size_mea / 2, size_mea / 2), size_mea, -size_mea, edgecolor='gray', fill=False))
    ax.scatter(0, 0, color='gray', s=scatter_size)
    # ax.scatter(-size_mea_um / 2, size_mea_um / 2, color='y', s=scatter_size)
    ax.scatter(x_aligned, y_aligned, color='g', s=scatter_size)
    ax.set_title(f"Frame: {fid}, Size: {size}")
    adjust_plot(ax, x_axis_range, y_axis_range, title='')
    return ax

