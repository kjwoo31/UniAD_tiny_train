
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data_custom.py nuscenes --root-path /mnt/OT_DATASET/dataset/nuscenes/data/250814_nuscenes_sample \
       --out-dir ./data/infos \
       --extra-tag nuscenes \
       --version v1.0-mini \
       --canbus ./data/nuscenes \