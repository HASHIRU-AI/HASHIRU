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
    
    def respond(message, chat_history):
        return model_manager.ask(message, chat_history)
    
    def user_message(msg: str, history: list) -> tuple[str, list]:
        """Adds user message to chat history"""
        history.append(gr.ChatMessage(role="user", content=msg))
        return "", history
    
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            avatar_images=("HASHIRU_2.png", "HASHIRU.png"),
            type="messages"
        )
        input_box = gr.Textbox()
        clear = gr.ClearButton([input_box, chatbot])

        def respond(message, chat_history):
            
            chat_history.append({
                "role":"user",
                "content":message
            })
            print("Chat history:", chat_history)
            chat_history = model_manager.run(chat_history)
            return "", chat_history

        msg_store = gr.State("")
        
        input_box.submit(
            lambda msg: (msg, msg, ""),  # Store message and clear input
            inputs=[input_box],
            outputs=[msg_store, input_box, input_box],
            queue=False
        ).then(
            user_message,  # Add user message to chat
            inputs=[msg_store, chatbot],
            outputs=[input_box, chatbot],
            queue=False
        ).then(
            model_manager.run,  # Generate and stream response
            inputs=[msg_store, chatbot],
            outputs=chatbot
        )

    demo.launch()
