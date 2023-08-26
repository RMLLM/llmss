from .gptj_modeling import GPTJForCausalLM
from .modeling_gpt_bigcode import GPTBigCodeForCausalLM

MODEL_REGISTRY = {
    "gptj": GPTJForCausalLM,
    "starcoder": GPTBigCodeForCausalLM,
}
