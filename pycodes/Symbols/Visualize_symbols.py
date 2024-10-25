from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots_adjust

from pycodes.modules.symbol_stimuli_utils import generate_symbol_frame


# Parameters
symbols = ['F', 'square', 'T', 'L', 'I']  # symbols to generate from available symbols in the function generate_symbol_frame

x_size_px = 768  # pixels
y_size_px = 768  # pixels

symbol_sizes_px = [150, 300, 500]  # pixels
symbol_centers = [(252, 300)]  # (x, y) in px assuming y-axis pointing down and x-axis pointing right

symbol_color = 1
background_color = 0


def main():

    nrows = len(symbol_sizes_px)*len(symbol_centers)
    ncols = len(symbols)
    dim = 4
    fontsize = 18
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*dim, nrows*dim))

    for i_ss, symbol_size in enumerate(symbol_sizes_px):
        for i_sc, symbol_center in enumerate(symbol_centers):
            i_row = i_ss * len(symbol_centers) + i_sc
            for i_x, symbol in enumerate(symbols):
                i_col = i_x
                frame = generate_symbol_frame(symbol, x_size_px, y_size_px,
                                              symbol_size, symbol_center,
                                              symbol_color, background_color)

                ax = axs[i_row, i_col]
                ax.set_title(f"{symbol}\n{symbol_size} px - {symbol_center}", fontsize=fontsize)
                ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
                ax.scatter(symbol_center[0], symbol_center[1], s=50, c='red', marker='o')
                ax.axis('off')
    subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()
    return


if __name__ == '__main__':
    main()
