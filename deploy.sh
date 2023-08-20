
python -m fastchat.serve.controller

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/lmsys_vicuna-13b-v1.3 --load-8bit

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/vicuna-13b-v1.5-16k --load-8bit

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/TheBloke_Wizard-Vicuna-30B-Uncensored-fp16 --load-8bit

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/Qwen_Qwen-7B-Chat

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/chatglm2-6b-32k

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/chatglm2-6b

python -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path models/Qwen-7B-Chat

python -m fastchat.serve.openai_api_server --host localhost --port 8000 

