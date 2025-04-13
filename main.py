from google.genai import types
from CEO.CEO import GeminiManager
from CEO.tool_loader import ToolLoader

if __name__ == "__main__":
    # Define the tool metadata for orchestration.
    # Load the tools using the ToolLoader class.
    tool_loader = ToolLoader()

    model_manager = GeminiManager(toolsLoader=tool_loader, gemini_model="gemini-2.0-flash")
    
    task_prompt = (
        "Give me a 3 stanza 4 line poem about drake vs kendrick lamar in old english style. Also, can you create a short story with the moral slow and steady wins the race?"
    )
    
    # Request a CEO response with the prompt.
    user_prompt_content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=task_prompt)],
    )
    response = model_manager.request([user_prompt_content])
    print("\nCEO Response:", response)
