from argparse import ArgumentParser

import redis
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Request(BaseModel):
    prompt: str
    max_new_tokens: int
    is_greedy: bool
    temperature: float
    top_p: float
    top_k: int


class Response(BaseModel):
    prompt: str
    continuation: str


def get_args():
    parser = ArgumentParser()
    producer_group = parser.add_argument_group("producer")
    producer_group.add_argument("--fastapi_host", type=str, default="127.0.0.1")
    producer_group.add_argument("--fastapi_port", type=int, default=8000)
    broker_group = parser.add_argument_group("broker")
    broker_group.add_argument("--redis_host", type=str, default="127.0.0.1")
    broker_group.add_argument("--redis_port", type=str, default=20000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # broker
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    producer_queue = "pqueue"
    subscriber_queue = "squeue"

    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        send_message = request.model_dump_json()
        redis_client.lpush(producer_queue, send_message)

        while True:
            if redis_client.llen(subscriber_queue):
                recv_message = redis_client.rpop(subscriber_queue)
                response = Response.model_validate_json(recv_message)
                break
        return response

    uvicorn.run(app, host=args.fastapi_host, port=args.fastapi_port)


if __name__ == "__main__":
    main()
