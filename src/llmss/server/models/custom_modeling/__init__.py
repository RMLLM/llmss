from .gpt_bigcode_modeling import GPTBigCodeForCausalLM
from .gptj_modeling import GPTJForCausalLM
from .llama_modeling import LlamaForCausalLM
from .open_llama_modeling import OpenLlamaForCausalLM

MODEL_REGISTRY = {"gptj": GPTJForCausalLM, 
                  "gpt_bigcode": GPTBigCodeForCausalLM, 
                  "llama": LlamaForCausalLM,
                  "open_llama": OpenLlamaForCausalLM}
