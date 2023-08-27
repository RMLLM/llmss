import json
from argparse import ArgumentParser

import redis
import torch
import torch.distributed as dist
from torch.nn.functional import softmax
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from llmss.server.models.custom_modeling import MODEL_REGISTRY
from llmss.server.models.utils.dist import initialize_torch_distributed
from llmss.server.models.utils.hub import weight_files
from llmss.server.models.utils.weights import Weights


def get_args():
    parser = ArgumentParser()
    consumer_group = parser.add_argument_group("consumer")
    consumer_group.add_argument("--model_type", type=str, required=True)
    consumer_group.add_argument("--pretrained_model_path", type=str, required=True)
    broker_group = parser.add_argument_group("broker")
    broker_group.add_argument("--redis_host", type=str, default="127.0.0.1")
    broker_group.add_argument("--redis_port", type=int, default=20000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    process_group, rank, _ = initialize_torch_distributed()  # process_group, rank, world_size

    if rank == 0:
        redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        producer_queue = "pqueue"
        subscriber_queue = "squeue"

    cuda_device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path,
            padding_side="left",
            truncation_side="left",
        )

    config = AutoConfig.from_pretrained(
        args.pretrained_model_path,
    )
    max_sequence_length = config.n_positions

    filenames = weight_files(args.pretrained_model_path)
    weights = Weights(filenames, device=cuda_device, dtype=dtype, process_group=process_group)

    model = MODEL_REGISTRY[args.model_type](config, weights)
    model.eval()

    torch.distributed.barrier(process_group)

    # NOTE (230818)
    # - 현재 단계에서는 batch 처리를 생각하지않음.
    # - pubsub 구조에서 subscriber가 model server인 형태
    # - subscriber를 여러개 사용할 때, 개별 subscriber를 batch를 처리하는 로직을 고민해봐야할 것 같음.
    #   - 애초에 고민하는 것 자체가 이상한 것일수도 있음.
    batch_size = 1

    while True:
        shared_payloads = [None]

        if rank == 0:
            if redis_client.llen(producer_queue):
                recv_message = redis_client.rpop(producer_queue)
                recv_payload = json.loads(recv_message)
                list_of_prompts = [recv_payload.pop("prompt")]
                max_new_tokens = recv_payload.pop("max_new_tokens")
                is_greedy = recv_payload.pop("is_greedy")
                temperature = recv_payload.pop("temperature")
                top_p = recv_payload.pop("top_p")
                top_k = recv_payload.pop("top_k")

                input_idss = tokenizer(
                    list_of_prompts,
                    return_attention_mask=False,
                    max_length=max_sequence_length,
                    padding=False,
                    truncation=True,
                )["input_ids"]

                shared_payloads = [
                    {
                        "input_idss": input_idss,
                        "max_new_tokens": max_new_tokens,
                        "is_greedy": is_greedy,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }
                ]

        dist.broadcast_object_list(shared_payloads, src=0)

        if shared_payloads == [None]:
            continue

        shared_payload = shared_payloads[0]
        buffer_idss = torch.tensor(shared_payload["input_idss"], device=cuda_device)
        buffer_idss_length = buffer_idss.size(-1)
        buffer_past_key_values = None

        if rank == 0:
            output_idss = torch.tensor([[] for _ in range(batch_size)], dtype=torch.int64).to(cuda_device)

        num_new_tokens = 0

        while num_new_tokens < shared_payload["max_new_tokens"]:
            with torch.no_grad():
                outputs = model(buffer_idss, past_key_values=buffer_past_key_values, use_cache=True)
                logits = outputs.logits
                buffer_past_key_values = outputs.past_key_values
                buffer_idss = torch.tensor([[0] for _ in range(batch_size)], dtype=torch.int64).to(cuda_device)

            if rank == 0:
                if shared_payload["is_greedy"]:
                    next_token_idss = logits[:, -1, :].argmax(-1).unsqueeze(-1)
                else:
                    list_of_warpers = []
                    if 0 < shared_payload["top_p"] < 1:
                        list_of_warpers.append(TopPLogitsWarper(shared_payload["top_p"]))
                    if shared_payload["top_k"] > 0:
                        list_of_warpers.append(TopKLogitsWarper(shared_payload["top_k"]))
                    if shared_payload["temperature"] != 1.0:
                        list_of_warpers.append(TemperatureLogitsWarper(shared_payload["temperature"]))
                    if not list_of_warpers:
                        logits_warpers = LogitsProcessorList(list_of_warpers)
                        next_token_scores = logits_warpers(buffer_idss, logits[:, -1, :])
                    else:
                        next_token_scores = logits[:, -1, :]
                    probs = softmax(next_token_scores, -1)
                    next_token_idss = torch.multinomial(probs, num_samples=1)

                buffer_idss = next_token_idss
                output_idss = torch.cat((output_idss, next_token_idss), dim=-1)
            buffer_idss_length += 1

            if buffer_idss_length > max_sequence_length:
                buffer_past_key_values = []

                for past_key_value in outputs.past_key_values:
                    buffer_past_key_values.append(
                        (
                            past_key_value[0][:, :, -max_sequence_length + 1 :, :],
                            past_key_value[1][:, :, -max_sequence_length + 1 :, :],
                        )
                    )
                buffer_past_key_values = tuple(buffer_past_key_values)

            dist.broadcast(buffer_idss, src=0)
            num_new_tokens += 1

        if rank == 0:
            list_of_continuations = tokenizer.batch_decode(output_idss.tolist())
            send_payload = {"prompt": list_of_prompts[0], "continuation": list_of_continuations[0]}  # batch_size = 1
            print(send_payload)
            send_message = json.dumps(send_payload, ensure_ascii=False)
            redis_client.lpush(subscriber_queue, send_message)


if __name__ == "__main__":
    main()
