#!/bin/bash

env

ARGS="api.py \
    --port $PORT0

echo $ARGS
python3 $ARGS
