import gradio as gr

import base64
from src.manager.manager import GeminiManager

model_manager = GeminiManager(gemini_model="gemini-2.0-flash")

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
    history.append({
        "role": "user",
        "content": message,
    })
    yield "", history
    for messages in model_manager.run(history):
        for message in messages:
            if message.get("role") == "summary":
                print(f"Summary: {message.get('content', '')}")
        yield "", messages


def update_model(model_name):
    print(f"Model changed to: {model_name}")
    pass


with gr.Blocks(css=css, fill_width=True, fill_height=True) as demo:
    with gr.Column(scale=1):
        with gr.Row(scale=0):
            gr.Markdown(_header_html)
            model_dropdown = gr.Dropdown(
                choices=[
                    "HASHIRU",
                    "Static-HASHIRU",
                    "Cloud-Only HASHIRU",
                    "Local-Only HASHIRU",
                    "No-Economy HASHIRU",
                ],
                value="HASHIRU",
                interactive=True,
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
            gr.ChatInterface(fn=run_model, type="messages", chatbot=chatbot,
                             additional_outputs=[chatbot], save_history=True)
if __name__ == "__main__":
    demo.launch()
