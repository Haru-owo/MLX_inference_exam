import os
import time
import re
import json
from mlx_lm import load
from mlx_lm.generate import stream_generate
from sampler_utils import make_sampler

# Root directory where all converted MLX models are stored
MODEL_ROOT = "./mlx_models"

# Global variables to hold the currently loaded model and tokenizer
model = None
tokenizer = None

def load_model(model_name):
    """
    Load an MLX model and tokenizer by name from the model root directory.

    Args:
        model_name (str): Name of the model directory inside ./mlx_models

    Returns:
        (model, tokenizer): Loaded model and tokenizer objects
    """
    global model, tokenizer
    model_path = os.path.join(MODEL_ROOT, model_name)
    model, tokenizer = load(model_path)
    return model, tokenizer

def load_generation_config(model_name):
    """
    Load the generation configuration (temperature, top_p, etc.) for a specific model.
    Looks for a 'generation_config.json' file inside the model directory.

    Args:
        model_name (str): Name of the model directory

    Returns:
        dict: Configuration dictionary. Returns empty dict if not found or invalid.
    """
    model_dir = os.path.join(MODEL_ROOT, model_name)
    config_path = os.path.join(model_dir, "generation_config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ {model_name} Failed to load config: {e}")
            return {}
    else:
        print(f"⚠️ No generation_config.json found in {model_name}.")
        return {}

def generate_with_instructions(user_input, model_name):
    """
    Generate a formatted prompt based on user input.
    If the input appears to be a math problem, apply a structured reasoning template.
    Otherwise, generate a simple detailed prompt.

    Args:
        user_input (str): The user's question or instruction
        model_name (str): The model directory name, used to load custom config

    Returns:
        str: A formatted prompt string for the model
    """
    generation_config = load_generation_config(model_name)

    # Custom XML-style tag for reasoning, e.g., <thought> ... </thought>
    reasoning_tag = generation_config.get("reasoning_tag", "thought")
    
    if is_math_problem(user_input):
        # If it seems like a math problem, use chain-of-thought format with \boxed{}
        prompt = f"""
        Question: {user_input}
        
        Please reason step by step, and put your final answer within \\boxed{{}}.

        Format your response as follows:
        <{reasoning_tag}>
        [Your reasoning steps here]
        </{reasoning_tag}>
        <answer>
        \\boxed{{[Your final answer here]}}
        </answer>
        """
    else:
        # For general questions, ask for a detailed response
        prompt = f"""
        Question: {user_input}

        Please provide a detailed answer.
        """
    return prompt

def is_math_problem(user_input):
    """
    Determine whether the input seems to be a math-related problem
    by checking for numbers and arithmetic operators.

    Args:
        user_input (str): The user's input string

    Returns:
        bool: True if input looks like a math problem, else False
    """
    math_keywords = r"[\d\+\-\*/=\(\)]"
    return bool(re.search(math_keywords, user_input))

def stream_response(model, tokenizer, messages, config):
    """
    Generate a streaming response from the model given chat history.

    Args:
        model: The MLX model object
        tokenizer: Tokenizer associated with the model
        messages (list[dict]): Chat history in message format (role + content)
        config (dict): Generation config (temperature, top_p, max_tokens, etc.)

    Yields:
        tuple: (token_text, status_string) where:
               - token_text is the next generated token (string)
               - status_string shows current tokens per second (e.g., "⏱ 12.45 tok/s")
    """
    # Create a single text prompt from chat messages using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # Create a sampler function based on temperature and top_p
    sampler_fn = make_sampler(
        temperature=config["temperature"],
        top_p=config["top_p"]
    )

    # For performance tracking: time and token count
    start_time = time.time()
    total_tokens = 0

    # Streaming token generation using MLX's stream_generate
    for token in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        sampler=sampler_fn,
        max_tokens=config["max_tokens"]
    ):
        total_tokens += 1
        elapsed = time.time() - start_time
        tps = total_tokens / elapsed if elapsed > 0 else 0

        # Yield the token text and generation speed info
        yield token, f"⏱ {tps:.2f} tok/s"
