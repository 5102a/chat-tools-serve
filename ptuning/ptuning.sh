#!/bin/bash

env

if [ "$show_board" == "True" ]; then
    echo "show_board"
    tensorboard --port $ARNOLD_TENSORBOARD_CURRENT_PORT --logdir $log_dir --bind_all &
fi

bash ./train-c10d.sh
