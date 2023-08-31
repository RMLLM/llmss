from .gptj_modeling import GPTJForCausalLM
from .gpt_bigcode_modeling import GPTBigCodeForCausalLM

MODEL_REGISTRY = {
    "gptj": GPTJForCausalLM,
    "gpt_bigcode": GPTBigCodeForCausalLM,
}
