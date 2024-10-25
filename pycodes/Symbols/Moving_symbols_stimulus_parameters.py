import numpy as np

# PARAMETERS TUNING
STIMULUS_VERSION_ID = "MovingSymbols_v1"
pixel_size = 3.5  # in µm per pixel

# - Frequency
stimulus_frequency = 50  # Hz

# - Stimulation area
SA_sq_dim = 768  # pixels
SA_x_size_px, SA_y_size_px = SA_sq_dim, SA_sq_dim  # pixels
SA_x_size_um, SA_y_size_um = SA_x_size_px * pixel_size, SA_y_size_px * pixel_size  # micrometers

# - Symbols
symbols = ['F']  # symbols to generate from available symbols in the function generate_symbol_frame
normalized_symbol_sizes = [0.14]  # normalized size of the symbol in the range [0, 1] (= ptg of the SA)
symbol_sizes_px = np.array([int(x*SA_sq_dim) for x in normalized_symbol_sizes])  # pixels
symbol_sizes_um = np.array([int(x*SA_sq_dim*pixel_size) for x in normalized_symbol_sizes])  # µm
symbol_color = 1
background_color = 0

# - Movement
# Trajectories are lists of (x, y) coordinates in the range [0, 1]
# corresponding to the normalized position of the symbol center on the screen.
symbol_trajectories = {
    "Trj1": [(0.5, 0.5), (0.6, 0.3), (0.5, 0.5), (0.6, 0.6), (0.3, 0.7), (0.3, 0.3)],
}
symbol_speeds = [50]  # pixels/s
symbol_speeds_um = [int(x*pixel_size) for x in symbol_speeds]  # µm/s
fixation_time = 0.25  # seconds
fixation_time_frames = int(fixation_time * stimulus_frequency)  # frames

# - Sequence to sequence transition
s2s_transition_time = 0.2  # seconds
s2s_transition_frames = int(s2s_transition_time * stimulus_frequency)  # frames


