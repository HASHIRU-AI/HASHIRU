import os
from dotenv import load_dotenv  
load_dotenv()
from src.manager import GeminiManager
from src.tool_manager import ToolManager
import gradio as gr
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
    HASHIRU AI - Runtime Test
  </span>
</div>
"""

# -------------------------------
# ToolManager Agent Creation
# -------------------------------
def create_agent_callback():
    print("\n[INFO] Creating agent using ToolManager...")
    manager = ToolManager()

    response = manager.runTool("AgentCreator", {
        "agent_name": "ui-runtime-agent",
        "system_prompt": "You answer questions.",
        "description": "Agent created via ToolManager",
        # No base_model passed — will trigger dynamic selection
    })

    print("[TOOL RESPONSE]", response)
    return response["message"]

# -------------------------------
# Gradio UI
# -------------------------------
if __name__ == "__main__":
    css = """
    #title-row { background: #2c2c2c; border-radius: 8px; padding: 8px; }
    """
    with gr.Blocks(css=css, fill_width=True, fill_height=True) as demo:
        with gr.Column():
            gr.HTML(_header_html)
            agent_create_button = gr.Button("🧪 Create Agent via ToolManager")
            result_output = gr.Textbox(label="Tool Response")

            agent_create_button.click(
                fn=create_agent_callback,
                inputs=[],
                outputs=[result_output]
            )

    demo.launch()
