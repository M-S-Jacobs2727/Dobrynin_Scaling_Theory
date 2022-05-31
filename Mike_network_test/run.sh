#!/usr/bin/bash
export TUNE_MAX_PENDING_TRIALS_PG=4
# CUBLAS_WORKSPACE_CONFIG=:16:8
python3 /proj/avdlab/projects/Solutions_ML/Dobrynin_Scaling_Theory/Mike_network_test/gen_and_train.py >/proj/avdlab/projects/Solutions_ML/mike_outputs/ray_tuning_no_pe.out