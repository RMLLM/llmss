from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmss.server.models.custom_modeling.gptj_modeling import GPTJForCausalLM
from llmss.server.models.utils.dist import initialize_torch_distributed
from llmss.server.models.utils.hub import weight_files
from llmss.server.models.utils.weights import Weights


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--is_greedy", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    process_group, rank, world_size = initialize_torch_distributed()
    test_ckpt_dirpath = Path("./temp_ckpt_for_test").absolute()

    if rank == 0:
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

    prompt = args.prompt
    if rank == 0:
        input_ids = tokenizer(prompt, return_attention_mask=False)["input_ids"]
        input_ids_len_cuda = torch.tensor(len(input_ids)).to(f"cuda:{rank}")
        input_ids_cuda = torch.tensor([input_ids]).to(f"cuda:{rank}")
    else:
        input_ids_len_cuda = torch.tensor(0).to(f"cuda:{rank}")

    dist.broadcast(input_ids_len_cuda, src=0)

    if rank != 0:
        input_ids_cuda = torch.tensor([0] * input_ids_len_cuda.item()).to(f"cuda:{rank}")

    dist.broadcast(input_ids_cuda, src=0)

    buffer_ids_cuda = input_ids_cuda
    buffer_ids_len_cuda = input_ids_len_cuda

    num_new_tokens = 0
    max_new_tokens = args.max_new_tokens
    is_greedy = args.is_greedy

    if rank == 0:
        output_ids = torch.tensor([[]], dtype=torch.int).to(f"cuda:{rank}")

    while num_new_tokens <= max_new_tokens:
        with torch.no_grad():
            logits = model(buffer_ids_cuda).logits

        if rank == 0:
            if is_greedy:
                next_token_id = logits[:, -1, :].argmax(-1).unsqueeze(0)
            else:
                m = Categorical(logits=logits[:, -1, :].squeeze())
                next_token_id = m.sample().reshape(-1, 1)

            output_ids = torch.cat((output_ids, next_token_id), dim=-1)
            buffer_ids_cuda = torch.cat((buffer_ids_cuda, next_token_id), dim=-1)
            buffer_ids_len_cuda += 1
            if buffer_ids_len_cuda.item() > max_sequence_length:
                buffer_ids_cuda = buffer_ids_cuda[:, max_sequence_length:]
                buffer_ids_len_cuda = torch.tensor(max_sequence_length).to(f"cuda:{rank}")

        dist.broadcast(buffer_ids_len_cuda, src=0)

        if rank != 0:
            buffer_ids_cuda = torch.tensor([0] * buffer_ids_len_cuda.item()).to(f"cuda:{rank}")
        dist.broadcast(buffer_ids_cuda, src=0)

        num_new_tokens += 1

        dist.barrier(process_group)

    if rank == 0:
        print(f"prompt: {args.prompt}")
        print(f"continuation: {tokenizer.decode(output_ids[0].tolist())}")


if __name__ == "__main__":
    main()
