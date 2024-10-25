# PARAMETERS TUNING
STIMULUS_VERSION_ID = "MovingSymbols_v1"

# - Stimulation area
SA_sq_dim = 768  # pixels
SA_x_size_px, SA_y_size_px = SA_sq_dim, SA_sq_dim  # pixels

# - Symbols
symbols = ['F']  # symbols to generate from available symbols in the function generate_symbol_frame
normalized_symbol_sizes = [0.18]  # normalized size of the symbol in the range [0, 1] (= ptg of the SA)
symbol_sizes_px = [int(x*SA_sq_dim) for x in normalized_symbol_sizes]  # pixels
symbol_color = 1
background_color = 0

# - Movement
# Trajectories are lists of (x, y) coordinates in the range [0, 1]
# corresponding to the normalized position of the symbol center on the screen.
symbol_trajectories = [
    [(0.5, 0.5), (0.1, 0.9), (0.5, 0.5), (0.9, 0.1), (0.75, 0.25), (0.25, 0.75)],  # Trajectory 1
]
symbol_speeds = [70]  # pixels/s


