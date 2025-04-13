from google.genai import types
from src.CEO import GeminiManager
from src.tool_loader import ToolLoader

if __name__ == "__main__":
    # Define the tool metadata for orchestration.
    # Load the tools using the ToolLoader class.
    tool_loader = ToolLoader()

    model_manager = GeminiManager(toolsLoader=tool_loader, gemini_model="gemini-2.0-flash")
    
    task_prompt = (
        "What is the peak price of trump coin in the last 30 days? "
        "Please provide the price in USD. "
    )
    
    # Request a CEO response with the prompt.
    # user_prompt_content = types.Content(
    #     role='user',
    #     parts=[types.Part.from_text(text=task_prompt)],
    # )
    # response = model_manager.request([user_prompt_content])
    response = model_manager.start_conversation()
