import gradio as gr
import os
import re
from sampler_utils import (
    load_generation_config,
    update_config_from_ui,
    save_generation_config,
    DEFAULT_CONFIG,
    PRESETS
)
from inference import load_model, stream_response, generate_with_instructions

# Directory where converted MLX models are stored
MODEL_ROOT = "./mlx_models"

# List all model directories available under ./mlx_models
model_dirs = sorted([
    d for d in os.listdir(MODEL_ROOT)
    if os.path.isdir(os.path.join(MODEL_ROOT, d))
])

# Load the first available model by default
current_model_name = model_dirs[0]
model, tokenizer = load_model(current_model_name)

# Load the saved generation configuration for the selected model
generation_config = load_generation_config(os.path.join(MODEL_ROOT, current_model_name))
slider_state = generation_config.copy()  # UI sliders will be initialized using this state

def split_reasoning_and_answer(text, tag=None):
    """
    Extracts reasoning and answer blocks from generated text using custom tags.

    Args:
        text (str): Full response text from the model
        tag (str): Custom tag for reasoning block, e.g. 'thought'

    Returns:
        (list, str): List of reasoning blocks and final answer string
    """
    if not tag:
        tag = generation_config.get("reasoning_tag", "thought")

    if tag and f"<{tag}>" in text:
        # Extract reasoning blocks enclosed in custom tags (e.g., <thought>...</thought>)
        pattern = rf"<{tag}>(.*?)</{tag}>"
        reasoning_blocks = re.findall(pattern, text, re.DOTALL)
        
        # Remove reasoning tags from original response to isolate the final answer
        answer = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return reasoning_blocks, answer
    else:
        # If no custom tag is found, treat the whole response as the answer
        return [], text.strip()

# Used to prevent input while the model is loading
loading_state = gr.State(False)

def load_model_and_update_ui(new_model_name):
    """
    Change the current model and update all UI components (sliders) accordingly.

    Args:
        new_model_name (str): The selected model name from dropdown

    Returns:
        status (str): Message about loading status
        Slider values: temperature, top_p, max_tokens, repetition_penalty
    """
    global model, tokenizer, generation_config, current_model_name, slider_state

    loading_message = f"⏳ '{new_model_name}' Loading the model..."

    # Free GPU memory by clearing previous model (if needed)
    model, tokenizer = None, None
    import gc
    gc.collect()

    # Load new model and config
    model, tokenizer = load_model(new_model_name)
    model_path = os.path.join(MODEL_ROOT, new_model_name)
    generation_config = load_generation_config(model_path)
    slider_state = generation_config.copy()
    current_model_name = new_model_name

    completion_message = f"✅ Model '{new_model_name}' loaded successfully!"
    return (
        completion_message,
        slider_state["temperature"],
        slider_state["top_p"],
        slider_state["max_tokens"],
        slider_state["repetition_penalty"]
    )

def update_config(temp, top_p, max_tokens, rep_penalty):
    """
    Update the generation configuration from slider inputs.

    Returns:
        str: Status message
    """
    global generation_config, slider_state
    generation_config = update_config_from_ui(temp, top_p, max_tokens, rep_penalty)
    slider_state.update(generation_config)
    return "🔧 Sampling settings have been updated."

def apply_preset(preset_name):
    """
    Apply one of the predefined generation presets.

    Args:
        preset_name (str): Name of the preset (e.g., "💡 Precision Mode")

    Returns:
        Status message + individual slider values to update the UI
    """
    preset = PRESETS[preset_name]
    return update_config(**preset), *preset.values()

def save_config():
    """
    Save the current slider settings to generation_config.json.

    Returns:
        str: Confirmation message
    """
    save_generation_config(os.path.join(MODEL_ROOT, current_model_name), slider_state)
    return "💾 Settings saved."

def stream_chat(user_input, history):
    """
    Generate a streaming response based on user input and chat history.

    Args:
        user_input (str): User's input message
        history (list): List of previous chat messages

    Yields:
        (str, list, str): Empty string (clears textbox), updated chat history, speed info
    """
    if loading_state.value:
        yield "The model is still loading. Please wait a moment...", history, ""
        return

    # Add user's message to chat history
    history.append({"role": "user", "content": user_input})
    partial_response = ""
    tag = generation_config.get("reasoning_tag", None)

    # Stream token-by-token response
    for token, tps in stream_response(model, tokenizer, history, generation_config):
        partial_response += token.text
        reasoning_blocks, answer = split_reasoning_and_answer(partial_response, tag=tag)

        # Build chat response from parsed blocks
        new_history = history[:-1] + [{"role": "user", "content": user_input}]

        if reasoning_blocks:
            new_history.append({
                "role": "assistant",
                "content": "🤔 Reasoning:\n" + "\n\n".join(reasoning_blocks)
            })

        if answer:
            new_history.append({
                "role": "assistant",
                "content": "✅ Answer:\n" + "\n\n" + answer
            })

        # Yield updated UI state
        yield "", new_history, tps


# === Gradio UI Construction ===

with gr.Blocks(title="🧠 MLX Inference") as demo:
    gr.Markdown("## 🤖 MLX Inference - LLM")

    # Dropdown to choose model
    model_selector = gr.Dropdown(choices=model_dirs, value=current_model_name, label="모델 선택")
    status = gr.Markdown()               # For messages (e.g. "model loaded")
    loading_overlay = gr.Markdown(visible=False)  # Not used currently

    # Expandable section for sampling settings
    with gr.Accordion("⚙️ Inference Settings", open=False):
        temp_slider = gr.Slider(0.1, 1.5, 0.05, label="🌡 Temperature", value=slider_state["temperature"])
        top_p_slider = gr.Slider(0.1, 1.0, 0.05, label="🎯 Top-p", value=slider_state["top_p"])
        max_tokens_slider = gr.Slider(64, 2048, 32, label="📏 Max Tokens", value=slider_state["max_tokens"])
        rep_penalty_slider = gr.Slider(1.0, 2.0, 0.05, label="🔁 Repetition Penalty", value=slider_state["repetition_penalty"])

        # Save button to persist slider values to config file
        gr.Button("💾 Save Settings").click(save_config, outputs=status)

        # Apply preset buttons
        with gr.Row():
            for name in PRESETS:
                gr.Button(name).click(
                    fn=lambda name=name: apply_preset(name),
                    outputs=[status, temp_slider, top_p_slider, max_tokens_slider, rep_penalty_slider]
                )

    # Chat UI section
    chatbot = gr.Chatbot(label="Chat Bot", type="messages")
    msg = gr.Textbox(placeholder="Type a message... (Press Enter to send)", show_label=False)
    clear = gr.Button("🧹 Reset Conversation")
    speed = gr.Markdown()  # Display token generation speed

    # Connect user message submission to streaming response
    msg.submit(stream_chat, [msg, chatbot], [msg, chatbot, speed])
    clear.click(lambda: [], None, chatbot)  # Clear chat on button press

    # When model selection changes, reload and update UI
    model_selector.change(
        load_model_and_update_ui,
        inputs=model_selector,
        outputs=[
            status, 
            temp_slider, 
            top_p_slider, 
            max_tokens_slider, 
            rep_penalty_slider
        ]
    )

    # Update generation config when sliders are moved
    for slider in [temp_slider, top_p_slider, max_tokens_slider, rep_penalty_slider]:
        slider.change(
            update_config,
            inputs=[temp_slider, top_p_slider, max_tokens_slider, rep_penalty_slider],
            outputs=status
        )

    # (Unused but present) UI overlay toggle for loading
    loading_overlay.change(
        lambda visible: "⏳ Loading model..." if visible else "",
        inputs=loading_overlay,
        outputs=loading_overlay
    )

# Launch the Gradio app
demo.launch()
