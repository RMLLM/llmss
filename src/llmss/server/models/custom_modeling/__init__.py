from .gpt_bigcode_modeling import GPTBigCodeForCausalLM
from .gptj_modeling import GPTJForCausalLM
from .llama_modeling import LlamaForCausalLM

MODEL_REGISTRY = {"gptj": GPTJForCausalLM, "gpt_bigcode": GPTBigCodeForCausalLM, "llama": LlamaForCausalLM}
