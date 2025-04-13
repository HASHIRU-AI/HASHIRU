from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from CEO.tool_loader import ToolLoader
from CEO.utils.suppress_outputs import suppress_output

class GeminiManager:
    def __init__(self, toolsLoader: ToolLoader, system_prompt_file="./models/system3.prompt", gemini_model="gemini-2.5-pro-exp-03-25"):
        load_dotenv()
        self.API_KEY = os.getenv("GEMINI_KEY")
        self.client = genai.Client(api_key=self.API_KEY)
        self.toolsLoader: ToolLoader = toolsLoader
        self.toolsLoader.load_tools()
        self.model_name = gemini_model
        with open(system_prompt_file, 'r', encoding="utf8") as f:
            self.system_prompt = f.read()

    def request(self, messages):
        try:
            response = suppress_output(self.client.models.generate_content)(
                #model='gemini-2.5-pro-preview-03-25',
                model=self.model_name,
                #model='gemini-2.5-pro-exp-03-25',
                #model='gemini-2.0-flash',
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.2,
                    tools=self.toolsLoader.getTools(),
                ),
            )
        except Exception as e:
            print(f"Error: {e}")
            shouldRetry = input("An error occurred. Do you want to retry? (y/n): ")
            if shouldRetry.lower() == "y":
                return self.request(messages)
            else:
                print("Ending the conversation.")
                return messages

        print(f"Response: {response}")
        
        if response.text is not None:
            assistant_content = types.Content(
                role='model' if self.model_name == "gemini-2.5-pro-exp-03-25" else 'assistant',
                parts=[types.Part.from_text(text=response.text)],
            )
            messages.append(assistant_content)
        if response.candidates[0].content:
            messages.append(response.candidates[0].content)
        if response.function_calls:
            parts = []
            for function_call in response.function_calls:
                toolResponse = None
                print(f"Function Name: {function_call.name}, Arguments: {function_call.args}")
                try:
                    toolResponse = self.toolsLoader.runTool(function_call.name, function_call.args)
                except Exception as e:
                    print(f"Error running tool: {e}")
                    toolResponse = {
                        "status": "error",
                        "message": f"Tool {function_call.name} failed to run.",
                        "output": str(e),
                    }
                print(f"Tool Response: {toolResponse}")
                tool_content = types.Part.from_function_response(
                        name=function_call.name,
                        response = {"result":toolResponse})
                try:
                    self.toolsLoader.load_tools()
                except Exception as e:
                    print(f"Error loading tools: {e}")
                    # delete the created tool
                    self.toolsLoader.delete_tool(function_call.name, toolResponse.tool_file_path)
                    tool_content = types.Part.from_function_response(
                            name=function_call.name,
                            response={"result":f"{function_call.name} with {function_call.args} doesn't follow the required format, please read the other tool implementations for reference." + str(e)})
                parts.append(tool_content)
            function_response_content = types.Content(
                role='model' if self.model_name == "gemini-2.5-pro-exp-03-25" else 'tool',
                parts=parts
            )
            messages.append(function_response_content)
            shouldContinue = input("Should I continue? (y/n): ")
            if shouldContinue.lower() == "y":
                return self.request(messages)
            else:
                print("Ending the conversation.")
                return messages
        else:
            print("No tool calls found in the response.")
            question = input("User: ")
            if ("exit" in question.lower() or "quit" in question.lower()):
                print("Ending the conversation.")
                return messages
            user_content = types.Content(
                role='user',
                parts=[types.Part.from_text(text=question)],
            )
            messages.append(user_content)
            return self.request(messages)