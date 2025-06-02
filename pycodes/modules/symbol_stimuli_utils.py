import matplotlib.pyplot as plt
import numpy as np

def symbol_location_check(symbol_center, symbol_size_px, frame_x_size_px, frame_y_size_px):
    """ Check if the symbol is within the frame boundaries.
        (0, 0) is the top-left corner of the frame.

        Args:
            symbol_center: symbol location in pixel coordinates [(x, y) assuming y-axis pointing down and x-axis pointing right]
            symbol_size_px: symbol dimension (square edge) in pixels
            frame_x_size_px, frame_y_size_px: frame dimensions in pixels
    """

    # Get corner coordinates of the symbol (in pixels)
    symbol_center_x = symbol_center[0]
    symbol_center_y = symbol_center[1]
    symbol_top_left_x, symbol_top_left_y = symbol_center_x - symbol_size_px // 2, symbol_center_y - symbol_size_px // 2
    symbol_top_right_x, symbol_top_right_y = symbol_top_left_x + symbol_size_px, symbol_top_left_y
    symbol_bottom_left_x, symbol_bottom_left_y = symbol_top_left_x, symbol_top_left_y + symbol_size_px
    symbol_bottom_right_x, symbol_bottom_right_y = symbol_top_right_x, symbol_bottom_left_y

    # Limits
    top_limit, bottom_limit, left_limit, right_limit = 0, frame_y_size_px, 0, frame_x_size_px

    # Check if the symbol is within the frame boundaries
    checks = [top_limit <= y < bottom_limit and left_limit <= x < right_limit for (x,y)
              in [(symbol_top_left_x, symbol_top_left_y), (symbol_top_right_x, symbol_top_right_y),
                  (symbol_bottom_left_x, symbol_bottom_left_y), (symbol_bottom_right_x, symbol_bottom_right_y)]]
    if not all(checks): raise ValueError("Symbol is out of frame boundaries")

    return symbol_center_x, symbol_center_y, symbol_top_left_x, symbol_top_left_y


def generate_symbol_frame(symbol,
                          frame_x_size_px, frame_y_size_px,
                          symbol_size_px, symbol_center, symbol_color, background_color,
                          digit_type=np.int8,
                          show=False):
    """ Generate a frame with a symbol with the specified size and location.
        The symbol is colored with the symbol_color and the frame is filled with the background color.

        Args:
        - symbol: any of the following symbols: 'F', 'square', 'T', 'L', 'I'
        - frame_x_size_px, frame_y_size_px: frame dimensions in pixels
        - symbol_size_px: symbol dimension (square edge) in pixels
        - symbol_center: symbol location in pixel coordinates [(x, y) assuming y-axis pointing down and x-axis pointing right]
        - symbol_color: color value to fill the symbol with
        - background_color: color value to fill the frame background with
        - digit_type: type of digit storage, to maximize data weight according to values (use 'float16' if symbol_color, background_color not int)
        - show: whether to show the frame

        Returns:
            frame: numpy array (frame_y_size_px, frame_x_size_px) with the generated frame

    """
    frame = np.full((frame_y_size_px, frame_x_size_px), background_color, dtype=digit_type)

    # Get coordinates of the symbol (in pixels)
    (symbol_center_x, symbol_center_y,
     symbol_top_left_x, symbol_top_left_y) = symbol_location_check(symbol_center, symbol_size_px, frame_x_size_px, frame_y_size_px)

    square_dim_row_start = int(symbol_top_left_y)
    square_dim_row_end = int(symbol_top_left_y + symbol_size_px)
    square_dim_col_start = int(symbol_top_left_x)
    square_dim_col_end = int(symbol_top_left_x + symbol_size_px)

    if symbol == 'F':
        set_of_params = [(symbol_color, (0, 0.2), (0, 1)),  # upper horizontal bar
                         (symbol_color, (0.4, 0.6), (0, 0.5)),  # central mid-horizontal bar
                         (symbol_color, (0, 1), (0, 0.2)),  # left vertical bar
                         ]

    elif symbol == 'E':
        set_of_params = [(symbol_color, (0, 0.2), (0, 1)),  # upper horizontal bar
                         (symbol_color, (0.4, 0.6), (0, 1)),  # central horizontal bar
                         (symbol_color, (0.8, 1), (0, 1)),  # lower horizontal bar
                         (symbol_color, (0, 1), (0, 0.2)),  # left vertical bar
                         ]

    elif symbol == 'E_hflip':
        set_of_params = [(symbol_color, (0, 0.2), (0, 1)),  # upper horizontal bar
                         (symbol_color, (0.4, 0.6), (0, 1)),  # central horizontal bar
                         (symbol_color, (0.8, 1), (0, 1)),  # lower horizontal bar
                         (symbol_color, (0, 1), (0.8, 1)),  # right vertical bar
                         ]

    elif symbol == 'E_p90deg':
        set_of_params = [(symbol_color, (0.8, 1), (0, 1)),  # lower horizontal bar
                         (symbol_color, (0, 1), (0, 0.2)),  # left vertical bar
                         (symbol_color, (0, 1), (0.4, 0.6)),  # central vertical bar
                         (symbol_color, (0, 1), (0.8, 1)),  # right vertical bar
                         ]

    elif symbol == 'E_m90deg':
        set_of_params = [(symbol_color, (0, 0.2), (0, 1)),  # upper horizontal bar
                         (symbol_color, (0, 1), (0, 0.2)),  # left vertical bar
                         (symbol_color, (0, 1), (0.4, 0.6)),  # central vertical bar
                         (symbol_color, (0, 1), (0.8, 1)),  # right vertical bar
                         ]

    elif symbol == 'square':
        set_of_params = [(symbol_color, (0, 0.2), (0, 1)),  # upper horizontal bar
                         (symbol_color, (0.8, 1), (0, 1)),  # lower horizontal bar
                         (symbol_color, (0, 1), (0, 0.2)),  # left vertical bar
                         (symbol_color, (0, 1), (0.8, 1)),  # right vertical bar
                         ]

    elif symbol == 'T':
        set_of_params = [(symbol_color, (0, 0.2), (0, 1)),  # upper horizontal bar
                         (symbol_color, (0, 1), (0.4, 0.6)),  # central vertical bar
                         ]

    elif symbol == 'L':
        set_of_params = [(symbol_color, (0.8, 1), (0, 1)),  # lower horizontal bar
                         (symbol_color, (0, 1), (0, 0.2)),  # left vertical bar
                         ]

    elif symbol == 'I':
        set_of_params = [(symbol_color, (0, 1), (0.4, 0.6)),  # central vertical bar
                         ]
    else:
        raise ValueError(f"Symbol {symbol} not recognized")

    for (color, hl, vl) in set_of_params:
        frame = add_h_bar(frame, color, square_dim_row_start, square_dim_row_end, square_dim_col_start, square_dim_col_end, h_lim=hl, v_lim=vl)

    # show_frame
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cax = ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        ax.scatter(symbol_center_x, symbol_center_y, s=50, c='red', marker='o')
        ax.scatter(symbol_top_left_x, symbol_top_left_y, s=50, c='blue', marker='o')
        ax.add_patch(plt.Rectangle((square_dim_col_start, square_dim_row_start), symbol_size_px, symbol_size_px, edgecolor='gray', facecolor='none'))
        fig.colorbar(cax)
        ax.grid(True, which='both', linestyle='-', linewidth=0.2)
        plt.show()
    return frame


def add_h_bar(frame, symbol_color,
              square_dim_row_start, square_dim_row_end, square_dim_col_start, square_dim_col_end,
              h_lim=(0,1), v_lim=(0, 1)):
    assert 0 <= v_lim[0] < 1 and 0 < v_lim[1] <= 1, "v_lim must be within [0,1]"
    v_dim = square_dim_col_end-square_dim_col_start
    v_start = int(square_dim_col_start + v_dim*v_lim[0])
    v_end = int(square_dim_col_start + v_dim*v_lim[1])

    assert 0 <= h_lim[0] < 1 and 0 < h_lim[1] <= 1, "h_lim must be within [0,1]"
    h_dim = square_dim_row_end-square_dim_row_start
    h_start = int(square_dim_row_start + h_dim*h_lim[0])
    h_end = int(square_dim_row_start + h_dim*h_lim[1])

    assert v_end-v_start > 0
    assert h_end-h_start > 0
    frame[h_start:h_end, v_start:v_end] = symbol_color

    return frame

