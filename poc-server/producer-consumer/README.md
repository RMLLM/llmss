# Producer-consumer style inference server
## Preliminary
Supported model checkpoints should be saved with safetensor format.

- Supported model
  - `gptj`, `gpt_bigcode` (e.g. starcoder), `llama`

```bash
pip install -r requirements.txt
```

## Launch
### Broker
```bash
# before install redis
# sudo apt-get install redis-server
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
# Require gpus along number of --nproc_per_node
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
consumer_server.py \
--pretrained_model_path ${SUPPORTED_MODEl_CHECKPOINT} \
--redis_host 127.0.0.1 \
--redis_port 20000
```

## Test
```bash
curl localhost:8000/generate \
  -X POST \
  -d '{"prompt":"What is deep learning?",
      "max_new_tokens":10,
      "is_greedy": false,
      "temperature": 1.0,
      "top_p": 0.9,
      "top_k": 50}' \
  -H 'Content-Type: application/json'
```
