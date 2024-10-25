import matplotlib.pyplot as plt
import numpy as np


def generate_symbol_frame(symbol,
                          x_size_px, y_size_px,
                          symbol_size_px, symbol_center, symbol_color, background_color,
                          show=False):
    """ Generate a frame with a symbol with the specified size and location.
        The symbol is colored with the symbol_color and the frame is filled with the background color.

        Args:
            symbol: any of the following symbols: 'F', 'square', 'T', 'L', 'I'
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

    elif symbol == 'square':
        bar_width = int(symbol_size_px * 0.2)
        # upper horizontal bar
        frame[square_dim_row_start:square_dim_row_start+bar_width, square_dim_col_start:square_dim_col_end] = symbol_color
        # lower horizontal bar
        frame[square_dim_row_end-bar_width:square_dim_row_end, square_dim_col_start:square_dim_col_end] = symbol_color
        # left vertical bar
        frame[square_dim_row_start:square_dim_row_end, square_dim_col_start:square_dim_col_start+bar_width] = symbol_color
        # right vertical bar
        frame[square_dim_row_start:square_dim_row_end, square_dim_col_end-bar_width:square_dim_col_end] = symbol_color

    elif symbol == 'T':
        bar_width = int(symbol_size_px * 0.2)
        # upper horizontal bar
        frame[square_dim_row_start:square_dim_row_start + bar_width,square_dim_col_start:square_dim_col_end] = symbol_color
        # central vertical bar
        frame[square_dim_row_start:square_dim_row_end,symbol_center_x - bar_width // 2:symbol_center_x + bar_width // 2] = symbol_color

    elif symbol == 'L':
        bar_width = int(symbol_size_px * 0.2)
        # lower horizontal bar
        frame[square_dim_row_end - bar_width:square_dim_row_end, square_dim_col_start:square_dim_col_end] = symbol_color
        # left vertical bar
        frame[square_dim_row_start:square_dim_row_end,square_dim_col_start:square_dim_col_start + bar_width] = symbol_color

    elif symbol == 'I':
        bar_width = int(symbol_size_px * 0.2)
        # central vertical bar
        frame[square_dim_row_start:square_dim_row_end, symbol_center_x-bar_width//2:symbol_center_x+bar_width//2] = symbol_color

    else:
        raise ValueError(f"Symbol {symbol} not recognized")

    # show_frame
    if show:
        fig, ax = plt.subplots(1,1, figsize=(5, 5))
        cax = ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        ax.scatter(symbol_center_x, symbol_center_y, s=50, c='red', marker='o')
        ax.scatter(symbol_top_left_x, symbol_top_left_y, s=50, c='blue', marker='o')
        ax.add_patch(plt.Rectangle((square_dim_col_start, square_dim_row_start), symbol_size_px, symbol_size_px, edgecolor='gray', facecolor='none'))
        fig.colorbar(cax)
        ax.grid(True, which='both', linestyle='-', linewidth=0.2)
        plt.show()
    return frame
