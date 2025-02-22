EVAL_DEFAULT_CONFIG = {
    "model": "vllm",
    # "model_hf_path": "meta-llama/Llama-3.2-1B-Instruct",
    "model_hf_path": "Qwen/Qwen2-1.5B",
    "num_fewshot": 8,
    "sampler": "min_p",
    "sampler_value": 0.1,
    "seed": 0,
    "task": "gsm8k_cot_llama",
    "temperature": 1.0,
}
