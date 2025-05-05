from google.genai import types
from src.manager import GeminiManager
from src.tool_manager import ToolManager
import gradio as gr
import time
import base64

_logo_bytes = open("HASHIRU_LOGO.png", "rb").read()
_logo_b64 = base64.b64encode(_logo_bytes).decode()
_header_html = f"""
<div style="
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-right: 24px;
">
  <img src="data:image/png;base64,{_logo_b64}" width="20" height="20" />
  <span style="margin-top: 8px; font-size: 20px; font-weight: bold; color: white;">
    HASHIRU AI
  </span>
</div>
"""

if __name__ == "__main__":
    model_manager = GeminiManager(gemini_model="gemini-2.0-flash")
    
    def user_message(msg: str, history: list) -> tuple[str, list]:
        """Adds user message to chat history"""
        history.append(gr.ChatMessage(role="user", content=msg))
        return "", history

    def handle_undo(history, undo_data: gr.UndoData):
        return history[:undo_data.index], history[undo_data.index]['content']
    
    def handle_retry(history, retry_data: gr.RetryData):
        new_history = history[:retry_data.index+1]
        # yield new_history, gr.update(interactive=False,)
        yield from model_manager.run(new_history)
        
    def handle_edit(history, edit_data: gr.EditData):
        new_history = history[:edit_data.index+1]
        new_history[-1]['content'] = edit_data.value
        # yield new_history, gr.update(interactive=False,)
        yield from model_manager.run(new_history)

    def update_model(model_name):
        print(f"Model changed to: {model_name}")
        pass
    
    css = """
    #title-row { background: #2c2c2c; border-radius: 8px; padding: 8px; }
    """
    with gr.Blocks(css=css, fill_width=True, fill_height=True) as demo:
        local_storage = gr.BrowserState(["", ""])
        with gr.Column(scale=1):
            with gr.Row(scale=0):
                gr.HTML(_header_html)
                model_dropdown = gr.Dropdown(
                        choices=[
                            "HASHIRU",
                            "Static-HASHIRU",
                            "Cloud-Only HASHIRU",
                            "Local-Only HASHIRU",
                            "No-Economy HASHIRU",
                        ],
                        value="HASHIRU",
                        # label="HASHIRU",
                        interactive=True,
                )

                model_dropdown.change(fn=update_model, inputs=model_dropdown, outputs=[])
            with gr.Row(scale=1):
                with gr.Sidebar(position="left"):
                    buttons = []
                    for i in range(1, 6):
                        button = gr.Button(f"Button {i}", elem_id=f"button-{i}")
                        button.click(fn=lambda x=i: print(f"Button {x} clicked"), inputs=[], outputs=[])
                        buttons.append(button)
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
                        type="messages",
                        show_copy_button=True,
                        editable="user",
                        scale=1,
                        render_markdown=True,
                    )
                    input_box = gr.Textbox(label="Chat Message", scale=0, interactive=True, submit_btn=True)
                    
                    chatbot.undo(handle_undo, chatbot, [chatbot, input_box])
                    chatbot.retry(handle_retry, chatbot, [chatbot, input_box])
                    chatbot.edit(handle_edit, chatbot, [chatbot, input_box])
                    
                    input_box.submit(
                        user_message,  # Add user message to chat
                        inputs=[input_box, chatbot],
                        outputs=[input_box, chatbot],
                        queue=False,
                    ).then(
                        model_manager.ask_llm,  # Generate and stream response
                        inputs=chatbot,
                        outputs=[chatbot, input_box],
                        queue=True,
                        show_progress="full",
                        trigger_mode="always_last"
                    )
    
    demo.launch()
