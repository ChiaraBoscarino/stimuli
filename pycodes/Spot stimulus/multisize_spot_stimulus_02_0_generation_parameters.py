# SET THE PARAMETERS HERE
SIZES = [150, 250, 600, 1200]  # diameter (microns)
TARGET_AREA_WIDTH = 1500  # microns
TARGET_AREA_HEIGHT = 1500  # microns

ON_DURATION = 1  # (seconds)
OFF_DURATION = 1  # (seconds)
INITIAL_ADAPTATION = 0  # (seconds)
STIMULUS_FREQUENCY = 40  # (frames/second)
BACKGROUND_COLOR_VALUE = 0
SPOT_COLOR_VALUE = 1

ARRANGEMENT = "radial"  # in ["grid", "radial"]
MAX_ADMITTED_ALIGNMENT_MARGIN_PTG = 0.45  # *100% of spot size
K = 8  # number of repetitions of each spot size

# STANDARD PARAMETERS
SPOT_STIMULUS_DIMENSION_px = 768
PIXEL_SIZE = 3.5
N_ELECTRODES = 16
ELECTRODE_SPACING = 100  # microns
MEA_SIZE_um = 1530  # N_ELECTRODES * ELECTRODE_SPACING
RIG_ID = 1  # 1 or 2 or 3

# Stimulus parameters
STIMULUS_NAME = "multisize_spot_tiling_stimulus"
VERSION = "V6_MEA1"
FOLDER_NAME = f"{VERSION}_{int(MAX_ADMITTED_ALIGNMENT_MARGIN_PTG*100)}error_{K}reps"
root = "C:\\Users\\chiar\\Documents\\rgc_typing"  # !! SET THIS for windows
# root = r'/home/idv-equipe-s8/Documents/GitHub/rgc_typing'  # !! SET THIS for linux


# COMPUTE DERIVED PARAMETERS
SPOT_STIMULUS_DIMENSION_um = SPOT_STIMULUS_DIMENSION_px * PIXEL_SIZE
SPOT_STIMULUS_CENTER_COORDINATE_um = SPOT_STIMULUS_DIMENSION_um / 2  # center of the stimulus (starting from the top left corner of the stimulus)
SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um = SPOT_STIMULUS_CENTER_COORDINATE_um - MEA_SIZE_um / 2  # top left corner of the target area (starting from the top left corner of the stimulus)
frame_dimensions = (SPOT_STIMULUS_DIMENSION_px, SPOT_STIMULUS_DIMENSION_px)

SPOT_SEQUENCE_DURATION_seconds = ON_DURATION + OFF_DURATION
SPOT_SEQUENCE_DURATION_frames = SPOT_SEQUENCE_DURATION_seconds * STIMULUS_FREQUENCY

STIM_DIR = f"{STIMULUS_NAME}_{FOLDER_NAME}"
fn = f"{STIMULUS_NAME}_{ARRANGEMENT}_error_{int(MAX_ADMITTED_ALIGNMENT_MARGIN_PTG * 100)}_{VERSION}"
coordinates_filename = f"{fn}_coordinates"
frame_stack_filename = f"{fn}_frame_stack"
file_bin = f"{frame_stack_filename}.bin"
vec_filename = f"{fn}"
spot_reference_filename = f"{fn}_spot_reference"
