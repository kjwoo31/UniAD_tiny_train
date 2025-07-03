#!/bin/bash

docker run \
-it --rm --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--gpus all \
--shm-size=8g \
-v /mnt/:/mnt/ \
uniad_torch1.12
