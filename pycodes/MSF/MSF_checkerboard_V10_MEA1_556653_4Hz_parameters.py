msf_stimulus_frequency = 4  # Hz

msf_n_blocks = 3
msf_C_per_block = 1  # CLASSIC CHECKERBOARD
msf_M_per_block = 1  # MSF CHECKERBOARD
msf_checkerboard_dim_px = 768
msf_checkerboard_sta_temporal_dimension = 4  # frames

# CLASSIC CHECKERBOARD SPECIFIC
cc_n_rep_seq = 1  # sequences
cc_n_non_rep_seq = 90  # sequences
cc_total_n_seq = cc_n_rep_seq + cc_n_non_rep_seq  # sequences
cc_id_rep_seq = 0
cc_n_frames_in_rep_seq = 60  # frames
cc_n_frames_in_non_rep_seq = 90  # frames
cc_full_sequence_length = cc_n_frames_in_rep_seq + cc_n_frames_in_non_rep_seq  # frames
cc_seq_per_block = 30  # sequences
cc_check_size_mum = 56  # µm

# MSF CHECKERBOARD SPECIFIC
msc_check_sizes_um_expected = [56, 112, 224, 448, 896, 1344]  # µm (list)

msc_n_rep_seq = 1  # sequences
msc_n_non_rep_seq = 90  # sequences
msc_total_n_seq = msc_n_rep_seq + msc_n_non_rep_seq  # sequences
msc_id_rep_seq = cc_id_rep_seq + cc_total_n_seq
msc_n_frames_in_rep_seq = 60  # frames
msc_n_frames_in_non_rep_seq = 90  # frames
msc_full_sequence_length = cc_n_frames_in_rep_seq + cc_n_frames_in_non_rep_seq  # frames
msc_seq_per_block = 30  # sequences

frames_per_C_block = cc_seq_per_block * cc_full_sequence_length  # frames
frames_per_M_block = msc_seq_per_block * msc_full_sequence_length  # frames
tot_frames_per_block = frames_per_C_block + frames_per_M_block  # frames
