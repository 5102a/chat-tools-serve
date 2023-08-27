import gc
from fastapi import FastAPI, Request
import uvicorn
import json
import datetime
import time as t
import torch
import argparse
from chat import agent_executor

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def torch_unload():
    global model, tokenizer
    model = tokenizer = None
    print(f'Memory cached: {torch.cuda.memory_reserved()}')
    print("清理显存")
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    print(f'Memory cached: {torch.cuda.memory_reserved()}')


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')

    start = t.time()
    response = model.run(query)
    end = t.time()
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        # "history": history,
        "status": 200,
        "time": time,
        "cost": end - start
    }
    log = "[" + time + "][" + str(end - start) + "]\n" + 'prompt:\n' + \
        query + '\nresponse:\n' + repr(response)
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()

    model = agent_executor

    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)
