EVAL_DEFAULT_CONFIG = {
    "model": "vllm",
    "model_hf_path": "Qwen/Qwen2.5-0.5B",
    "num_fewshot": 8,
    "sampler": "min_p",
    "sampler_value": 0.1,
    "seed": 0,
    "task": "gsm8k_cot_llama",
    "temperature": 1.0,
}

MODELS_NICE_NAMES_DICT = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-v0.1": "Mistral-7B",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B",
}

MODELS_ORDER_LIST = [
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
    "Mistral-7B-v0.1",
]

MODELS_TYPE_DICT = {
    "Qwen/Qwen2.5-0.5B": "Base",
    "Qwen/Qwen2.5-0.5B-Instruct": "Instruct",
    "Qwen/Qwen2.5-1.5B": "Base",
    "Qwen/Qwen2.5-1.5B-Instruct": "Instruct",
    "Qwen/Qwen2.5-3B": "Base",
    "Qwen/Qwen2.5-3B-Instruct": "Instruct",
    "Qwen/Qwen2.5-7B": "Base",
    "Qwen/Qwen2.5-7B-Instruct": "Instruct",
    "mistralai/Mistral-7B-v0.1": "Base",
    "mistralai/Mistral-7B-Instruct-v0.1": "Instruct",
}

MODELS_TYPE_ORDER_LIST = [
    "Base",
    "Instruct",
]

SAMPLERS_NICE_NAMES_DICT = {
    "none": "None",
    "min_p": "Min-P",
    "top_p": "Top-P",
}

SAMPLERS_ORDER_LIST = [
    "None",
    "Min-P",
    "Top-P",
]

TASKS_ORDER_LIST = [
    "gsm8k_cot_llama",
]
