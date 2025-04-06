import json
import argparse
from huggingface_hub import login, snapshot_download
from mlx_lm.convert import convert
import os

def load_hf_token(json_path: str) -> str:
    """
    Load the Hugging Face API token from a JSON file.

    Args:
        json_path (str): Path to the JSON file (e.g., "token.json")

    Returns:
        str: Hugging Face access token
    """
    with open(json_path, "r") as f:
        return json.load(f)["hf_token"]

def convert_model(model_name: str, mlx_output_dir: str, hf_token: str, quantize: bool):
    """
    Download a Hugging Face model and convert it to MLX format.

    Args:
        model_name (str): Model repo ID on Hugging Face (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        mlx_output_dir (str): Output directory for the converted MLX model
        hf_token (str): Hugging Face access token
        quantize (bool): Whether to apply 4-bit quantization (True) or not (False)
    """
    # Authenticate to Hugging Face Hub
    login(token=hf_token)

    print("📦 Downloading model from Hugging Face...")
    # Download model snapshot to a local cache directory
    hf_model_path = snapshot_download(repo_id=model_name, token=hf_token)

    print(f"🔄 Converting to MLX format {'with quantization' if quantize else 'without quantization'}...")
    # Convert the Hugging Face model to MLX format
    convert(
        hf_path=hf_model_path,     # Input: path to downloaded HF model
        mlx_path=mlx_output_dir,   # Output: directory to save MLX model
        quantize=quantize          # Apply 4-bit quantization if requested
    )

def main():
    """
    Command-line entry point for downloading and converting a Hugging Face model to MLX.
    Usage examples:
        python install.py deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        python install.py LGAI-EXAONE/EXAONE-Deep-7.8B --quantize
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Download and convert a Hugging Face model to MLX format.")
    parser.add_argument("model_name", help="Hugging Face model repo ID (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    parser.add_argument("--quantize", action="store_true", help="Apply 4-bit quantization to the model")
    args = parser.parse_args()

    # Load Hugging Face token from local token.json
    token_file = "token.json"
    hf_token = load_hf_token(token_file)

    model_name = args.model_name
    # Extract the final part of the model name (used for output directory)
    model_basename = model_name.split("/")[-1]
    # Add "-int4" suffix if quantization is enabled
    suffix = "-int4" if args.quantize else ""
    # Final output directory path
    output_dir = os.path.join("mlx_models", model_basename + suffix)

    print("🔑 Loading Hugging Face token...")
    print(f"⬇️ Preparing to download and convert model: {model_name}")
    print(f"📁 Output directory: {output_dir}")

    # Perform the download and conversion process
    convert_model(model_name, output_dir, hf_token, args.quantize)

    print(f"✅ Done! Model saved to: {output_dir}")

# This block makes the script executable directly via CLI
if __name__ == "__main__":
    main()
