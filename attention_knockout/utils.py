import torch
from nnsight import LanguageModel

def load_model(name: str):
    """
    Load a pre-trained model.
    """
    if name == 'gemma-2-2b':
        model = LanguageModel(
            'google/gemma-2-2b',
            device_map='cuda:0',
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True
        )
    elif name == 'gemma-2-9b':
        model = LanguageModel(
            'google/gemma-2-9b',
            device_map='cuda:0',
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True
        )
    elif name == 'llama1b':
        model = LanguageModel(
            'meta-llama/Llama-3.2-1B',
            device_map='cuda:0',
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True
        )
    else:
        raise ValueError(f"Model '{name}' not supported.")
    
    return model