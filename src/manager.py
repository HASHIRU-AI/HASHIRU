from google import genai
from google.genai import types
from google.genai.types import *
import os
from dotenv import load_dotenv
import sys
from src.tool_loader import ToolLoader
from src.utils.suppress_outputs import suppress_output
import logging
import gradio as gr

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
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
        self.messages = []

    def generate_response(self, messages):
        return self.client.models.generate_content(
            model=self.model_name,
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
            logger.info(
                f"Function Name: {function_call.name}, Arguments: {function_call.args}")
            try:
                toolResponse = self.toolsLoader.runTool(
                    function_call.name, function_call.args)
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
                response={"result": toolResponse})
            try:
                self.toolsLoader.load_tools()
            except Exception as e:
                logger.info(f"Error loading tools: {e}. Deleting the tool.")
                # delete the created tool
                self.toolsLoader.delete_tool(
                    toolResponse['output']['tool_name'], toolResponse['output']['tool_file_path'])
                tool_content = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": f"{function_call.name} with {function_call.args} doesn't follow the required format, please read the other tool implementations for reference." + str(e)})
            parts.append(tool_content)
        return {
            "role": "tool",
            "content": repr(types.Content(
                    role='model' if self.model_name == "gemini-2.5-pro-exp-03-25" else 'tool',
                    parts=parts
            ))
        }

    def format_chat_history(self, messages=[]):
        formatted_history = []
        for message in messages:
            # Skip thinking messages (messages with metadata)
            if not (message.get("role") == "assistant" and "metadata" in message):
                role = "model"
                parts = [types.Part.from_text(text=message.get("content", ""))]
                match message.get("role"):
                    case "user":
                        role = "user"
                    case "tool":
                        role = "tool"
                        formatted_history.append(
                            eval(message.get("content", "")))
                        continue
                    case "function_call":
                        role = "model"
                        formatted_history.append(
                            eval(message.get("content", "")))
                        continue
                    case _:
                        role = "model"
                formatted_history.append(types.Content(
                    role=role,
                    parts=parts
                ))
        return formatted_history

    def run(self, messages):
        print("Messages: ", messages)
        chat_history = self.format_chat_history(messages)
        logger.debug(f"Chat history: {chat_history}")
        try:
            response = suppress_output(self.generate_response)(chat_history)
        except Exception as e:
            logger.debug(f"Error generating response: {e}")
            messages.append({
                "role": "assistant",
                "content": f"Error generating response: {e}"
            })
            logger.error(f"Error generating response: {e}")
            yield messages, gr.update(interactive=True)
            return
        logger.debug(f"Response: {response}")
        print("Response: ", response)

        if (not response.text and not response.function_calls):
            messages.append({
                "role": "assistant",
                "content": "No response from the model.",
                "metadata": {"title": "No response from the model."}
            })

        # Attach the llm response to the messages
        if response.text is not None and response.text != "":
            messages.append({
                "role": "assistant",
                "content": response.text
            })
            yield messages, gr.update(interactive=False,)

        # Attach the function call response to the messages
        if response.candidates[0].content and response.candidates[0].content.parts:
            # messages.append(response.candidates[0].content)
            messages.append({
                "role": "function_call",
                "content": repr(response.candidates[0].content),
            })

        # Invoke the function calls if any and attach the response to the messages
        if response.function_calls:
            calls = self.handle_tool_calls(response)
            messages.append(calls)
            yield from self.run(messages)
            return
        print("Final messages: ", messages)
        yield messages, gr.update(interactive=True)
