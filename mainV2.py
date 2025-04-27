from google.genai import types
from src.manager import GeminiManager
from src.tool_loader import ToolLoader
import gradio as gr
import time

if __name__ == "__main__":
    # Define the tool metadata for orchestration.
    # Load the tools using the ToolLoader class.
    tool_loader = ToolLoader()

    model_manager = GeminiManager(toolsLoader=tool_loader, gemini_model="gemini-2.0-flash")
    
    def user_message(msg: str, history: list) -> tuple[str, list]:
        """Adds user message to chat history"""
        history.append(gr.ChatMessage(role="user", content=msg))
        return "", history

    def handle_undo(history, undo_data: gr.UndoData):
        return history[:undo_data.index], history[undo_data.index]['content']
    
    def handle_retry(history, retry_data: gr.RetryData):
        new_history = history[:retry_data.index]
        yield from model_manager.run(new_history)
        
    def handle_edit(history, edit_data: gr.EditData):
        new_history = history[:edit_data.index]
        new_history[-1]['content'] = edit_data.value
        return new_history
    
    with gr.Blocks(fill_width=True, fill_height=True) as demo:
        gr.Markdown("# Hashiru AI")
        
        chatbot = gr.Chatbot(
            avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
            type="messages",
            show_copy_button=True,
            editable="user",
            scale=1
        )
        input_box = gr.Textbox(max_lines=5, label="Chat Message", scale=0)
        
        chatbot.undo(handle_undo, chatbot, [chatbot, input_box])
        chatbot.retry(handle_retry, chatbot, chatbot)
        chatbot.edit(handle_edit, chatbot, chatbot)
        
        input_box.submit(
            user_message,  # Add user message to chat
            inputs=[input_box, chatbot],
            outputs=[input_box, chatbot],
            queue=False,
        ).then(
            model_manager.run,  # Generate and stream response
            inputs=chatbot,
            outputs=[chatbot, input_box],
            queue=True,
            show_progress="full",
            trigger_mode="always_last"
        )

    demo.launch(share=True)
