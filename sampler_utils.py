import os
import json
import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate

# Default generation configuration
DEFAULT_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 512,
    "repetition_penalty": 1.1
}

# Preset modes for generation (e.g., precision vs creativity)
PRESETS = {
    "💡 Precision Mode": {
        "temperature": 0.4,
        "top_p": 0.85,
        "max_tokens": 512,
        "repetition_penalty": 1.2
    },
    "🎨 Creative Mode": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 512,
        "repetition_penalty": 1.0
    }
}

def load_generation_config(model_dir: str) -> dict:
    """
    Load generation configuration from a JSON file inside the model directory.
    Falls back to the DEFAULT_CONFIG if the file does not exist or is invalid.
    """
    config_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            # Merge user config with default config (user values override defaults)
            return {**DEFAULT_CONFIG, **user_config}
        except:
            # If the file is corrupted or invalid, use default
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_generation_config(model_dir, config: dict):
    """
    Save the generation configuration as a JSON file in the model directory.
    """
    config_path = os.path.join(model_dir, "generation_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def update_config_from_ui(temp, top_p, max_tokens, rep_penalty) -> dict:
    """
    Update configuration dictionary with values provided from a UI or CLI.
    All values should be validated before being passed in.
    """
    return {
        "temperature": temp,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
        "repetition_penalty": rep_penalty
    }

def make_sampler(temperature=1.0, top_p=1.0):
    """
    Create a sampling function to generate the next token based on model logits.

    - If temperature is 0, use greedy decoding (argmax).
    - If temperature > 0, apply top-p nucleus sampling with softmax.
    """
    if temperature == 0.0:
        # Greedy sampling (pick highest probability token)
        return lambda logits: mx.argmax(logits, axis=-1)

    def sampler(logits: mx.array) -> mx.array:
        """
        Top-p (nucleus) sampling implementation:
        1. Scale logits by temperature.
        2. Apply softmax to get probabilities.
        3. Sort tokens by probability and keep the smallest set with cumulative prob >= top_p.
        4. Normalize the selected probabilities.
        5. Randomly sample from the filtered token set.
        """
        logits = logits.astype(mx.float32) / temperature
        probs = np.array(mx.softmax(logits, axis=-1)).flatten()

        sorted_indices = np.argsort(-probs)  # descending order
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Determine how many top tokens to keep (cumulative prob >= top_p)
        cutoff = np.searchsorted(cumulative_probs, top_p)
        filtered_indices = sorted_indices[:cutoff + 1]
        filtered_probs = probs[filtered_indices]
        filtered_probs /= filtered_probs.sum()  # normalize

        # Randomly select one token based on the filtered distribution
        sampled_token = np.random.choice(filtered_indices, p=filtered_probs)
        return mx.array([int(sampled_token)], dtype=mx.int32)

    return sampler

def generate_response(model, tokenizer, prompt, config: dict) -> str:
    """
    Generate a text response using a model, tokenizer, and sampling config.

    Args:
        model: The MLX LLM model.
        tokenizer: Tokenizer compatible with the model.
        prompt: Input string to generate from.
        config: Generation parameters (temperature, top_p, etc.)

    Returns:
        Generated string as model output.
    """
    # Create the sampling function with given parameters
    sampler_fn = make_sampler(
        temperature=config["temperature"],
        top_p=config["top_p"]
    )

    # Run the generate function using MLX LLM interface
    return generate(
        model,
        tokenizer,
        prompt,
        sampler=sampler_fn,
        max_tokens=config["max_tokens"]
    )
