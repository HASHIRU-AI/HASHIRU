from typing import List
import gradio as gr

import base64
from src.manager.manager import GeminiManager, Mode
from enum import Enum

_logo_bytes = open("HASHIRU_LOGO.png", "rb").read()
_logo_b64 = base64.b64encode(_logo_bytes).decode()
_header_html = f"""
<div style="
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: flex-start;
    width: 30%;
">
  <img src="data:image/png;base64,{_logo_b64}" width="40" class="logo"/>
  <h1>
    HASHIRU AI
  </h1>
</div>
"""
css = """
    .logo { margin-right: 20px; }
    """


def run_model(message, history):
    if 'text' in message:
        history.append({
            "role": "user",
            "content": message['text']
        })
    if 'files' in message:
        for file in message['files']:
            history.append({
                "role": "user",
                "content": (file,)
            })
    yield "", history
    for messages in model_manager.run(history):
        yield "", messages


with gr.Blocks(css=css, fill_width=True, fill_height=True) as demo:
    model_manager = GeminiManager(
        gemini_model="gemini-2.0-flash", modes=[mode for mode in Mode])

    def update_model(modeIndexes: List[int]):
        modes = [Mode(i+1) for i in modeIndexes]
        print(f"Selected modes: {modes}")
        model_manager.set_modes(modes)

    with gr.Column(scale=1):
        with gr.Row(scale=0):
            gr.Markdown(_header_html)
            model_dropdown = gr.Dropdown(
                choices=[mode.name for mode in Mode],
                value=model_manager.get_current_modes,
                interactive=True,
                type="index",
                multiselect=True,
                label="Select Modes",
            )

            model_dropdown.change(
                fn=update_model, inputs=model_dropdown, outputs=[])
        with gr.Row(scale=1):
            chatbot = gr.Chatbot(
                avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
                type="messages",
                show_copy_button=True,
                editable="user",
                scale=1,
                render_markdown=True,
                placeholder="Type your message here...",
            )
            gr.ChatInterface(fn=run_model,
                             type="messages",
                             chatbot=chatbot,
                             additional_outputs=[chatbot],
                             save_history=True,
                             editable=True,
                             multimodal=True,)
if __name__ == "__main__":
    demo.launch()
