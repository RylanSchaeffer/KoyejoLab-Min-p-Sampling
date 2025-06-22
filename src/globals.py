EVAL_DEFAULT_CONFIG = {
    "model": "vllm",
    "model_hf_path": "google/gemma-2-2b",
    # "model_hf_path": "google/gemma-2-2b-it",
    # "model_hf_path": "meta-llama/Llama-3.1-8B",
    # "model_hf_path": "meta-llama/Llama-3.2-3B-Instruct",
    # "model_hf_path": "mistralai/Mistral-7B-v0.1",
    # "model_hf_path": "Qwen/Qwen2.5-0.5B",
    "num_fewshot": 5,
    # "num_fewshot": 8,
    # "sampler": "min_p",
    "sampler": "basic",
    "sampler_value": 0.0,
    # "sampler_value": 0.1,
    "seed": 0,
    # "task": "gpqa_main_generative_n_shot",
    # "task": "gsm8k_cot",
    # "task": "gsm8k_cot_llama",
    # "task": "hendrycks_math_algebra",
    "task": "mmlu_pro_biology",
    "temperature": 1.0,
}

MODELS_NICE_NAMES_DICT = {
    "Qwen/Qwen2.5-0.5B": "Qwen 2.5 0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen 2.5 0.5B Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen 2.5 1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 2.5 1.5B Instruct",
    "Qwen/Qwen2.5-3B": "Qwen 2.5 3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5 3B Instruct",
    "Qwen/Qwen2.5-7B": "Qwen 2.5 7B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B Instruct",
    "mistralai/Mistral-7B-v0.1": "Mistral 7Bv0.1",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7Bv0.1 Instruct",
    "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B Instruct",
    "meta-llama/Llama-3.1-8B": "Llama 3.1 8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "google/gemma-2-2b": "Gemma 2 2B",
    "google/gemma-2-2b-it": "Gemma 2 2B Instruct",
    "google/gemma-2-9b": "Gemma 2 9B",
    "google/gemma-2-9b-it": "Gemma 2 9B Instruct",
}

MODELS_ORDER_LIST = [
    "Qwen 2.5 0.5B",
    "Qwen 2.5 0.5B Instruct",
    "Qwen 2.5 1.5B",
    "Qwen 2.5 1.5B Instruct",
    "Qwen 2.5 3B",
    "Qwen 2.5 3B Instruct",
    "Qwen 2.5 7B",
    "Qwen 2.5 7B Instruct",
    "Mistral 7Bv0.1",
    "Mistral 7Bv0.1 Instruct",
    "Llama 3.2 3B",
    "Llama 3.2 3B Instruct",
    "Llama 3.1 8B",
    "Llama 3.1 8B Instruct",
    "Gemma 2 2B",
    "Gemma 2 2B Instruct",
    "Gemma 2 9B",
    "Gemma 2 9B Instruct",
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
    "meta-llama/Llama-3.2-3B": "Base",
    "meta-llama/Llama-3.2-3B-Instruct": "Instruct",
    "meta-llama/Llama-3.1-8B": "Base",
    "meta-llama/Llama-3.1-8B-Instruct": "Instruct",
    "google/gemma-2-2b": "Base",
    "google/gemma-2-2b-it": "Instruct",
    "google/gemma-2-9b": "Base",
    "google/gemma-2-9b-it": "Instruct",
}

MODELS_TYPE_ORDER_LIST = [
    "Base",
    "Instruct",
]

SAMPLERS_NICE_NAMES_DICT = {
    "basic": "Basic",
    # In earlier version of the code, we called basic sampling "standard sampling" and then decided to rename it
    # to "basic" because "standard" has a presumption of normalcy that we don't want to imply.
    "standard": "Basic",
    "min_p": "Min-p",
    "top_p": "Top-p",
    "top_k": "Top-k",
}

SAMPLERS_ORDER_LIST = [
    "Basic",
    "Top-p",
    "Top-k",
    "Min-p",
]

TASK_NICE_NAMES_DICT = {
    "gpqa_main_generative_n_shot": "GPQA",
    "gsm8k_cot_llama": "GSM8K CoT Llama",
    "gsm8k_cot": "GSM8K CoT",
    "hendrycks_math_algebra": "MATH Algebra",
    "hendrycks_math_counting_and_prob": "MATH Counting and Probability",
    "hendrycks_math_geometry": "MATH Geom",
    "hendrycks_math_intermediate_algebra": "MATH Intermediate Algebra",
    "hendrycks_math_num_theory": "MATH Num Theory",
    "hendrycks_math_prealgebra": "MATH Prealgebra",
    "hendrycks_math_precalc": "MATH Precalc",
    "mmlu_pro_biology": "MMLU Pro Biology",
    "mmlu_pro_business": "MMLU Pro Business",
    "mmlu_pro_chemistry": "MMLU Pro Chemistry",
    "mmlu_pro_computer_science": "MMLU Pro Computer Science",
    "mmlu_pro_economics": "MMLU Pro Economics",
    "mmlu_pro_engineering": "MMLU Pro Engineering",
    "mmlu_pro_health": "MMLU Pro Health",
    "mmlu_pro_history": "MMLU Pro History",
    "mmlu_pro_law": "MMLU Pro Law",
    "mmlu_pro_math": "MMLU Pro Math",
    "mmlu_pro_other": "MMLU Pro Other",
    "mmlu_pro_philosophy": "MMLU Pro Philosophy",
    "mmlu_pro_physics": "MMLU Pro Physics",
    "mmlu_pro_psychology": "MMLU Pro Psychology",
}

TASKS_ORDER_LIST = [
    "GSM8K CoT",
]
