#!/bin/bash

env

python3 -m fastchat.serve.controller &
python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port $PORT0 &
python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path $model_path
