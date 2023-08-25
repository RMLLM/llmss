# Producer-consumer style inference server
## Preliminary
https://huggingface.co/heegyu/kogpt-j-350m/tree/main 의 모델을 사용해 테스트 하였습니다. safetensor의 형태로 checkpoint가 저장되어있어야합니다.

```bash
pip install -r requirements.txt
```

## Launch
### Broker
```bash
# before install redis
redis-server --port 20000
```

### Producer
```bash
python producer_server.py \
--fastapi_host 127.0.0.1 \
--fastapi_port 8000 \
--redis_host 127.0.0.1 \
--redis_port 20000
```

### Consumer
```bash
# Currently, only support gptj
# Require gpus along number of --nproc_per_node
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
consumer_server.py \
--model_type gptj \
--pretrained_model_path /data/nick_262/llmss/temp_ckpt_for_test \
--redis_host 127.0.0.1 \
--redis_port 20000
```

## Test
```bash
curl \
-X 'POST' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
    "prompt": "이것은 테스트입니다.",
    "max_new_tokens": 32,
    "is_greedy": false,
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50
}' \
'http://localhost:8000/generate'
```
