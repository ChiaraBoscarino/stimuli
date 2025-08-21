""" Code to generate the checkerboard stack sequences upon rescaling (pixel --> check)
    if not already generated with stimulus generation scripts.
"""
import os
import numpy as np
from pycodes.modules import stimulus_utils


# --> SETUP HERE PARAMS, PATHS TO STIMULI and MSF VERSION
stimuli_dir = r'I:\STIMULI'
from pycodes.MSF.MSF_checkerboard_V10_MEA1_556653_4Hz_parameters import *
msf_dir = os.path.join(stimuli_dir,"MSF", "MSF_checkerboard_V10_MEA1_4Hz")
pixel_size = 3.5  # µm

# AUTOMATIC SETUP
msf_files_dir = os.path.join(msf_dir, "files")
cc_rep_seq_stim_stack_fp = os.path.join(msf_files_dir,  "cc_sequence_0.npy")
msf_rep_seq_stim_stack_fp = os.path.join(msf_files_dir, "msc_sequence_0.npy")
cc_checkerboard_rep_sequence_sta_fp = os.path.join(msf_dir, "cc_chk_rep.npy")
msc_checkerboard_rep_sequence_sta_fp = os.path.join(msf_dir, "msc_chk_rep.npy")

# CODE
cc_check_size = int(round(cc_check_size_mum / pixel_size))
msc_check_sizes_px, msc_check_sizes_counts = stimulus_utils.get_msc_sizes(str(msf_files_dir))  # px (list)
msc_check_sizes_um = np.array(msc_check_sizes_px) * pixel_size  # µm
assert np.array_equal(msc_check_sizes_um, msc_check_sizes_um_expected), "The expected and the retrieved check sizes are different"
msc_common_size = np.gcd.reduce(msc_check_sizes_px.astype(int))

cc_chk_rep = stimulus_utils.get_checkerboard_sequence(cc_rep_seq_stim_stack_fp, rescale=True,
                                                      n_pixel_per_check_old=cc_check_size,
                                                      n_pixel_per_check_new=1)
msc_chk_rep = stimulus_utils.get_checkerboard_sequence(msf_rep_seq_stim_stack_fp, rescale=True,
                                                       n_pixel_per_check_old=msc_common_size,
                                                       n_pixel_per_check_new=1)

np.save(cc_checkerboard_rep_sequence_sta_fp, np.array(cc_chk_rep))
np.save(msc_checkerboard_rep_sequence_sta_fp, np.array(msc_chk_rep))

print(f'CC: {cc_chk_rep.shape}')
print(f'\tsaved to {cc_checkerboard_rep_sequence_sta_fp}')
print(f'MSF: {msc_chk_rep.shape}')
print(f'\tsaved to {msc_checkerboard_rep_sequence_sta_fp}')
