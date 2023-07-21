from argparse import ArgumentParser
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from torch.nn.functional import softmax
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from llmss.server.models.custom_modeling.gptj_modeling import GPTJForCausalLM
from llmss.server.models.utils.dist import initialize_torch_distributed
from llmss.server.models.utils.hub import weight_files
from llmss.server.models.utils.weights import Weights


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=str, nargs="+", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--is_greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0, help="If you don't wanna use this, set to 1.0")
    parser.add_argument("--top_p", type=float, default=0.95, help="If you don't wanna use this, set to 1.0")
    parser.add_argument("--top_k", type=int, default=50, help="If you don't wanna use this, set to 0")
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    assert args.max_new_tokens > 0, f"Value of max_new_tokens should be over than 0."
    assert 0.0 < args.temperature <= 1.0, f"Value of temperature is not valid."
    assert 0.0 < args.top_p <= 1.0, f"Value of top_p is not valid."
    assert args.top_k >= 0, f"Value of top_k is not valid."

    process_group, rank, world_size = initialize_torch_distributed()
    test_ckpt_dirpath = Path("./temp_ckpt_for_test").absolute()

    if rank == 0:
        start_time = time()
        if not test_ckpt_dirpath.exists():
            tokenizer = AutoTokenizer.from_pretrained("heegyu/kogpt-j-350m")
            model = AutoModelForCausalLM.from_pretrained("heegyu/kogpt-j-350m")
            tokenizer.save_pretrained(test_ckpt_dirpath)
            model.save_pretrained(test_ckpt_dirpath, safe_serialization=True)

    torch.distributed.barrier(process_group)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        test_ckpt_dirpath,
        padding_side="left",
        truncation_side="left",
    )

    config = AutoConfig.from_pretrained(
        test_ckpt_dirpath,
    )
    max_sequence_length = config.n_positions

    torch.distributed.barrier(process_group)
    filenames = weight_files(test_ckpt_dirpath)
    weights = Weights(filenames, device=device, dtype=dtype, process_group=process_group)

    model = GPTJForCausalLM(config, weights)
    model.eval()

    list_of_prompts = args.prompts
    batch_size = len(list_of_prompts)

    if rank == 0:
        input_idss = tokenizer(
            list_of_prompts,
            return_attention_mask=False,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(f"cuda:{rank}")
    else:
        input_idss = torch.zeros((batch_size, max_sequence_length), dtype=torch.int64).to(f"cuda:{rank}")

    dist.broadcast(input_idss, src=0)

    valid_max_sequence_length = input_idss.ne(tokenizer.pad_token_id).sum(-1).max()
    input_idss = input_idss[:, -valid_max_sequence_length.item() :]

    buffer_idss = input_idss

    num_new_tokens = 0
    max_new_tokens = args.max_new_tokens
    is_greedy = args.is_greedy

    if rank == 0:
        output_idss = torch.tensor([[] for _ in range(batch_size)], dtype=torch.int64).to(f"cuda:{rank}")

    if args.use_cache:
        buffer_past_key_values = None

        while num_new_tokens < max_new_tokens:
            with torch.no_grad():
                outputs = model(buffer_idss, past_key_values=buffer_past_key_values, use_cache=True)
                logits = outputs.logits
                buffer_past_key_values = outputs.past_key_values
                buffer_idss = torch.tensor([[0] for _ in range(batch_size)], dtype=torch.int64).to(f"cuda:{rank}")

            if rank == 0:
                if is_greedy:
                    next_token_idss = logits[:, -1, :].argmax(-1).unsqueeze(-1)
                else:
                    list_of_warpers = []
                    if 0 < args.top_p < 1:
                        list_of_warpers.append(TopPLogitsWarper(args.top_p))
                    if args.top_k > 0:
                        list_of_warpers.append(TopKLogitsWarper(args.top_k))
                    if args.temperature != 1.0:
                        list_of_warpers.append(TemperatureLogitsWarper(args.temperature))
                    if not list_of_warpers:
                        logits_warpers = LogitsProcessorList(list_of_warpers)
                        next_token_scores = logits_warpers(buffer_idss, logits[:, -1, :])
                    else:
                        next_token_scores = logits[:, -1, :]
                    probs = softmax(next_token_scores, -1)
                    next_token_idss = torch.multinomial(probs, num_samples=1)

                buffer_idss = next_token_idss
                output_idss = torch.cat((output_idss, next_token_idss), dim=-1)
            valid_max_sequence_length += 1

            if valid_max_sequence_length > max_sequence_length:
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
    else:
        while num_new_tokens < max_new_tokens:
            with torch.no_grad():
                outputs = model(buffer_idss, use_cache=False)
                logits = outputs.logits

            if rank == 0:
                if is_greedy:
                    next_token_idss = logits[:, -1, :].argmax(-1).unsqueeze(-1)
                else:
                    list_of_warpers = []
                    if 0 < args.top_p < 1:
                        list_of_warpers.append(TopPLogitsWarper(args.top_p))
                    if args.top_k > 0:
                        list_of_warpers.append(TopKLogitsWarper(args.top_k))
                    if args.temperature != 1.0:
                        list_of_warpers.append(TemperatureLogitsWarper(args.temperature))
                    if not list_of_warpers:
                        logits_warpers = LogitsProcessorList(list_of_warpers)
                        next_token_scores = logits_warpers(buffer_idss, logits[:, -1, :])
                    else:
                        next_token_scores = logits[:, -1, :]
                    probs = softmax(next_token_scores, -1)
                    next_token_idss = torch.multinomial(probs, num_samples=1)

                output_idss = torch.cat((output_idss, next_token_idss), dim=-1)
                buffer_idss = torch.cat((buffer_idss, next_token_idss), dim=-1)
                valid_max_sequence_length += 1

                if valid_max_sequence_length > max_sequence_length:
                    buffer_idss = buffer_idss[:, -max_sequence_length:]
                    valid_max_sequence_length = torch.tensor(max_sequence_length).to(f"cuda:{rank}")

            dist.broadcast(valid_max_sequence_length, src=0)

            if rank != 0:
                buffer_idss = torch.zeros((batch_size, valid_max_sequence_length.item()), dtype=torch.int64).to(
                    f"cuda:{rank}"
                )
            dist.broadcast(buffer_idss, src=0)

            num_new_tokens += 1

    if rank == 0:
        end_time = time()
        print(f"elapsed time: {end_time - start_time}")
        print(f"prompts: {list_of_prompts}")
        print(f"continuations: {[tokenizer.decode(output_ids) for output_ids in output_idss]}")


if __name__ == "__main__":
    main()
