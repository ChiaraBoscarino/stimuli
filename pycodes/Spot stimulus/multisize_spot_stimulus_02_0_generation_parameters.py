# SET THE PARAMETERS HERE
SIZES = [150, 250, 600, 1200]  # diameter (microns)
TARGET_AREA_WIDTH = 500  # microns
TARGET_AREA_HEIGHT = 500  # microns

ON_DURATION = 1  # (seconds)
OFF_DURATION = 1  # (seconds)
INITIAL_ADAPTATION = 0  # (seconds)
STIMULUS_FREQUENCY = 4  # (frames/second)
BACKGROUND_COLOR_VALUE = 0
SPOT_COLOR_VALUE = 1

ARRANGEMENT = "radial"  # in ["grid", "radial"]
MAX_ADMITTED_ALIGNMENT_MARGIN_PTG = 0.3  # *100% of spot size
K = 12  # number of repetitions of each spot size

# STANDARD PARAMETERS
SPOT_STIMULUS_DIMENSION_px = 768
PIXEL_SIZE = 3.5
N_ELECTRODES = 16
ELECTRODE_SPACING = 30  # microns
MEA_SIZE_um = 458  # (N_ELECTRODES -1) * ELECTRODE_SPACING + ELECTRODE_DIAMETER
RIG_ID = 1  # 1 or 2 or 3

# Stimulus parameters
STIMULUS_NAME = "MSSpots"
VERSION = "V7_MEA1"
STIM_ID = f"{STIMULUS_NAME}_{VERSION}_{int(MAX_ADMITTED_ALIGNMENT_MARGIN_PTG*100)}error_{K}reps_{STIMULUS_FREQUENCY}Hz"
stimuli_dir = '/home/idv-equipe-s8/Documents/GitHub/hiOsight/stimuli/MultisizeSpots'


# COMPUTE DERIVED PARAMETERS
SPOT_STIMULUS_DIMENSION_um = SPOT_STIMULUS_DIMENSION_px * PIXEL_SIZE
SPOT_STIMULUS_CENTER_COORDINATE_um = SPOT_STIMULUS_DIMENSION_um / 2  # center of the stimulus (starting from the top left corner of the stimulus)
SPOT_STIMULUS_TARGET_AREA_ORIGIN_COORDINATE_um = SPOT_STIMULUS_CENTER_COORDINATE_um - MEA_SIZE_um / 2  # top left corner of the target area (starting from the top left corner of the stimulus)
frame_dimensions = (SPOT_STIMULUS_DIMENSION_px, SPOT_STIMULUS_DIMENSION_px)

SPOT_SEQUENCE_DURATION_seconds = ON_DURATION + OFF_DURATION
SPOT_SEQUENCE_DURATION_frames = SPOT_SEQUENCE_DURATION_seconds * STIMULUS_FREQUENCY

coordinates_filename = f"{STIM_ID}_coordinates"
frame_stack_filename = f"{STIM_ID}_frame_stack"
file_bin = f"{frame_stack_filename}.bin"
vec_filename = f"{STIM_ID}"
spot_reference_filename = f"{STIM_ID}_spot_reference"
