from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import sys
from src.tool_loader import ToolLoader
from src.utils.suppress_outputs import suppress_output
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

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

    def generate_response(self, messages):
        return self.client.models.generate_content(
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
    
    def handle_tool_calls(self, response):
        parts = []
        for function_call in response.function_calls:
            toolResponse = None
            logger.info(f"Function Name: {function_call.name}, Arguments: {function_call.args}")
            try:
                toolResponse = self.toolsLoader.runTool(function_call.name, function_call.args)
            except Exception as e:
                logger.warning(f"Error running tool: {e}")
                toolResponse = {
                    "status": "error",
                    "message": f"Tool {function_call.name} failed to run.",
                    "output": str(e),
                }
            logger.debug(f"Tool Response: {toolResponse}")
            tool_content = types.Part.from_function_response(
                    name=function_call.name,
                    response = {"result":toolResponse})
            try:
                self.toolsLoader.load_tools()
            except Exception as e:
                logger.info(f"Error loading tools: {e}. Deleting the tool.")
                # delete the created tool
                self.toolsLoader.delete_tool(toolResponse['output']['tool_name'], toolResponse['output']['tool_file_path'])
                tool_content = types.Part.from_function_response(
                        name=function_call.name,
                        response={"result":f"{function_call.name} with {function_call.args} doesn't follow the required format, please read the other tool implementations for reference." + str(e)})
            parts.append(tool_content)
        return types.Content(
            role='model' if self.model_name == "gemini-2.5-pro-exp-03-25" else 'tool',
            parts=parts
        )
    
    def run(self, messages):
        try:
            response = suppress_output(self.generate_response)(messages)
        except Exception as e:
            logger.debug(f"Error generating response: {e}")
            shouldRetry = input("An error occurred. Do you want to retry? (y/n): ")
            if shouldRetry.lower() == "y":
                return self.run(messages)
            else:
                print("Ending the conversation.")
                return messages

        logger.debug(f"Response: {response}")
        
        if (not response.text and not response.function_calls):
            print("No response from the model.")
        
        # Attach the llm response to the messages
        if response.text is not None:
            print("CEO:", response.text)
            assistant_content = types.Content(
                role='model' if self.model_name == "gemini-2.5-pro-exp-03-25" else 'assistant',
                parts=[types.Part.from_text(text=response.text)],
            )
            messages.append(assistant_content)
            
        # Attach the function call response to the messages
        if response.candidates[0].content and response.candidates[0].content.parts:
            messages.append(response.candidates[0].content)
            
        # Invoke the function calls if any and attach the response to the messages
        if response.function_calls:
            messages.append(self.handle_tool_calls(response))
            shouldContinue = input("Should I continue? (y/n): ")
            if shouldContinue.lower() == "y":
                return self.run(messages)
            else:
                print("Ending the conversation.")
                return messages
        else:
            logger.debug("No tool calls found in the response.")
            # Start the loop again
            return self.start_conversation(messages)
    
    def start_conversation(self, messages=[]):
        question = input("User: ")
        if ("exit" in question.lower() or "quit" in question.lower()):
            print("Ending the conversation.")
            return messages
        user_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=question)],
        )
        messages.append(user_content)
        
        # Start the conversation loop
        return self.run(messages)