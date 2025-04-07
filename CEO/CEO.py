from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from CEO.tool_loader import ToolLoader

class GeminiManager:
    def __init__(self, toolsLoader: ToolLoader, system_prompt_file="./models/system2.prompt"):
        load_dotenv()
        self.API_KEY = os.getenv("GEMINI_KEY")
        self.client = genai.Client(api_key=self.API_KEY)
        self.toolsLoader = toolsLoader
        self.toolsLoader.load_tools()
        with open(system_prompt_file, 'r', encoding="utf8") as f:
            self.system_prompt = f.read()

    def request(self, messages):
        response = self.client.models.generate_content(
            model='gemini-2.5-pro-preview-03-25',
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,
                tools=self.toolsLoader.getTools(),
            ),
        )

        print(f"Response: {response}")
        
        if response.text is not None:
            assistant_content = types.Content(
                role='assistant',
                parts=[types.Part.from_text(text=response.text)],
            )
            messages.append(assistant_content)
            if "EOF" in response.text:
                return messages
        if response.candidates[0].content:
            messages.append(response.candidates[0].content)
        if response.function_calls:
            parts = []
            for function_call in response.function_calls:
                toolResponse = None
                print(f"Function Name: {function_call.name}, Arguments: {function_call.args}")
                toolResponse = self.toolsLoader.runTool(function_call.name, function_call.args)
                print(f"Tool Response: {toolResponse}")
                tool_content = types.Part.from_function_response(
                        name=function_call.name,
                        response = {"result":toolResponse})
                try:
                    self.toolsLoader.load_tools()
                except Exception as e:
                    print(f"Error loading tools: {e}")
                    tool_content = types.Part.from_function_response(
                            name=function_call.name,
                            response={"result":"New tool doesn't follow the required format, please read the other tool implementations for reference." + str(e)})
                parts.append(tool_content)
            function_response_content = types.Content(
                role='tool', parts=parts
            )
            messages.append(function_response_content)
            self.request(messages)
        else:
            print("No tool calls found in the response.")
            return messages