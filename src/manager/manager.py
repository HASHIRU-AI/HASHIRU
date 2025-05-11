from google import genai
from google.genai import types
from google.genai.types import *
import os
from dotenv import load_dotenv
import sys
from src.manager.tool_manager import ToolManager
from src.manager.utils.suppress_outputs import suppress_output
import logging
import gradio as gr

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class GeminiManager:
    def __init__(self, toolsLoader: ToolManager = None,
                 system_prompt_file="./src/models/system3.prompt",
                 gemini_model="gemini-2.5-pro-exp-03-25",
                 local_only=False, allow_tool_creation=True,
                 cloud_only=False, use_economy=True):
        load_dotenv()
        self.toolsLoader: ToolManager = toolsLoader
        if not toolsLoader:
            self.toolsLoader: ToolManager = ToolManager()

        self.local_only = local_only
        self.allow_tool_creation = allow_tool_creation
        self.cloud_only = cloud_only
        self.use_economy = use_economy

        self.API_KEY = os.getenv("GEMINI_KEY")
        self.client = genai.Client(api_key=self.API_KEY)
        self.toolsLoader.load_tools()
        self.model_name = gemini_model
        with open(system_prompt_file, 'r', encoding="utf8") as f:
            self.system_prompt = f.read()
        self.messages = []

    def generate_response(self, messages):
        tools = self.toolsLoader.getTools()
        return self.client.models.generate_content(
            model=self.model_name,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,
                tools=tools,
            ),
        )

    def handle_tool_calls(self, response):
        parts = []
        i = 0
        for function_call in response.function_calls:
            title = ""
            thinking = ""
            toolResponse = None
            logger.info(
                f"Function Name: {function_call.name}, Arguments: {function_call.args}")
            title = f"Invoking `{function_call.name}` with `{function_call.args}`\n"
            yield {
                "role": "assistant",
                "content": thinking,
                "metadata": {
                    "title": title,
                    "id": i,
                    "status": "pending",
                }
            }
            try:
                toolResponse = self.toolsLoader.runTool(
                    function_call.name, function_call.args)
            except Exception as e:
                logger.warning(f"Error running tool: {e}")
                toolResponse = {
                    "status": "error",
                    "message": f"Tool `{function_call.name}` failed to run.",
                    "output": str(e),
                }
            logger.debug(f"Tool Response: {toolResponse}")
            thinking += f"Tool responded with ```\n{toolResponse}\n```\n"
            yield {
                "role": "assistant",
                "content": thinking,
                "metadata": {
                    "title": title,
                    "id": i,
                    "status": "done",
                }
            }
            tool_content = types.Part.from_function_response(
                name=function_call.name,
                response={"result": toolResponse})
            try:
                self.toolsLoader.load_tools()
            except Exception as e:
                logger.info(f"Error loading tools: {e}. Deleting the tool.")
                thinking += f"Error loading tools: {e}. Deleting the tool.\n"
                yield {
                    "role": "assistant",
                    "content": thinking,
                    "metadata": {
                        "title": title,
                        "id": i,
                        "status": "done",
                    }
                }
                # delete the created tool
                self.toolsLoader.delete_tool(
                    toolResponse['output']['tool_name'], toolResponse['output']['tool_file_path'])
                tool_content = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": f"{function_call.name} with {function_call.args} doesn't follow the required format, please read the other tool implementations for reference." + str(e)})
            parts.append(tool_content)
            i += 1
        yield {
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
            yield messages
            return
        logger.debug(f"Response: {response}")

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
            yield messages

        # Attach the function call response to the messages
        if response.candidates[0].content and response.candidates[0].content.parts:
            # messages.append(response.candidates[0].content)
            messages.append({
                "role": "function_call",
                "content": repr(response.candidates[0].content),
            })

        # Invoke the function calls if any and attach the response to the messages
        if response.function_calls:
            for call in self.handle_tool_calls(response):
                yield messages + [call]
                if (call.get("role") == "tool" 
                    or (call.get("role") == "assistant" and call.get("metadata", {}).get("status") == "done")):
                    messages.append(call)
            yield from self.run(messages)
            return
        yield messages
