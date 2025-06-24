#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis_tools/visualize/run.py \
 --predroot /home/labuser/bjyang/UniAD_train/UniAD/data/results_tiny_e2e.pkl  \
 --out_folder ./output/tiny_e2e_img_videos/  \
 --demo_video tiny_e2e_torch1.12_438-552.avi --project_to_cam True