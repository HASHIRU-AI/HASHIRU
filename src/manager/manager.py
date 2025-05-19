from enum import Enum, auto
from typing import List
from google import genai
from google.genai import types
from google.genai.types import *
import os
from dotenv import load_dotenv
import sys
from src.manager.agent_manager import AgentManager
from src.manager.budget_manager import BudgetManager
from src.manager.tool_manager import ToolManager
from src.manager.utils.suppress_outputs import suppress_output
import logging
import gradio as gr
from sentence_transformers import SentenceTransformer
import torch
from src.tools.default_tools.memory_manager import MemoryManager
from pathlib import Path
from google.genai.errors import APIError
import backoff
import mimetypes
import json

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Mode(Enum):
    ENABLE_AGENT_CREATION = auto()
    ENABLE_LOCAL_AGENTS = auto()
    ENABLE_CLOUD_AGENTS = auto()
    ENABLE_TOOL_CREATION = auto()
    ENABLE_TOOL_INVOCATION = auto()
    ENABLE_RESOURCE_BUDGET = auto()
    ENABLE_ECONOMY_BUDGET = auto()
    ENABLE_MEMORY = auto()

def format_tool_response(response, indent=0):
    """
    Format a tool response for display with proper handling of newlines in string values.
    This preserves the dictionary structure while making string values with newlines display properly.
    """
    indent_str = "  " * indent
    result = []
    
    if isinstance(response, dict):
        result.append("{")
        items = list(response.items())
        for i, (key, value) in enumerate(items):
            end_comma = "" if i == len(items) - 1 else ","
            formatted_value = format_tool_response(value, indent + 1)
            result.append(f"{indent_str}  \"{key}\": {formatted_value}{end_comma}")
        result.append(f"{indent_str}}}")
        return "\n".join(result)
    
    elif isinstance(response, list):
        result.append("[")
        for i, item in enumerate(response):
            end_comma = "" if i == len(response) - 1 else ","
            formatted_item = format_tool_response(item, indent + 1)
            result.append(f"{indent_str}  {formatted_item}{end_comma}")
        result.append(f"{indent_str}]")
        return "\n".join(result)
    
    elif isinstance(response, str):
        # Handle multiline strings by using triple quotes and preserving newlines
        if "\n" in response:
            # Replace newlines with actual newlines and proper indentation
            lines = response.split("\n")
            indented_lines = [f"{indent_str}    {line}" for line in lines]
            joined_lines = "\n".join(indented_lines)
            return f"'''\n{joined_lines}\n{indent_str}  '''"
        else:
            # Regular string
            return f"\"{response}\""
    
    elif response is None:
        return "null"
    
    elif isinstance(response, (int, float, bool)):
        return str(response).lower() if isinstance(response, bool) else str(response)
    
    else:
        # Fallback for other types
        return f"\"{str(response)}\""
class GeminiManager:
    def __init__(self, system_prompt_file="./src/models/system4.prompt",
                 gemini_model="gemini-2.5-pro-exp-03-25",
                 modes: List[Mode] = []):
        load_dotenv()
        self.budget_manager = BudgetManager()

        self.toolsLoader: ToolManager = ToolManager()

        self.agentManager: AgentManager = AgentManager()

        self.API_KEY = os.getenv("GEMINI_KEY")
        self.client = genai.Client(api_key=self.API_KEY)
        self.model_name = gemini_model
        self.memory_manager = MemoryManager()
        with open(system_prompt_file, 'r', encoding="utf8") as f:
            self.system_prompt = f.read()
        self.messages = []
        self.set_modes(modes)

    def get_current_modes(self):
        return [mode.name for mode in self.modes]

    def set_modes(self, modes: List[Mode]):
        self.modes = modes
        self.budget_manager.set_resource_budget_status(
            self.check_mode(Mode.ENABLE_RESOURCE_BUDGET))
        self.budget_manager.set_expense_budget_status(
            self.check_mode(Mode.ENABLE_ECONOMY_BUDGET))
        self.toolsLoader.set_creation_mode(
            self.check_mode(Mode.ENABLE_TOOL_CREATION))
        self.toolsLoader.set_invocation_mode(
            self.check_mode(Mode.ENABLE_TOOL_INVOCATION))
        self.agentManager.set_creation_mode(
            self.check_mode(Mode.ENABLE_AGENT_CREATION))
        self.agentManager.set_local_invocation_mode(
            self.check_mode(Mode.ENABLE_LOCAL_AGENTS))
        self.agentManager.set_cloud_invocation_mode(
            self.check_mode(Mode.ENABLE_CLOUD_AGENTS))

    def check_mode(self, mode: Mode):
        return mode in self.modes

    @backoff.on_exception(backoff.expo,
                          APIError,
                          max_tries=3,
                          jitter=None)
    def generate_response(self, messages):
        tools = self.toolsLoader.getTools()
        return self.client.models.generate_content_stream(
            model=self.model_name,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,
                tools=tools,
            ),
        )

    def handle_tool_calls(self, function_calls):
        parts = []
        i = 0
        for function_call in function_calls:
            title = ""
            thinking = ""
            toolResponse = None
            logger.info(
                f"Function Name: {function_call.name}, Arguments: {function_call.args}")
            title = f"Invoking `{function_call.name}` with \n```json\n{format_tool_response(function_call.args)}\n```\n"
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
            thinking += f"Tool responded with \n```json\n{format_tool_response(toolResponse)}\n```\n"
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
                if function_call.name == "ToolCreator" or function_call.name == "ToolDeletor":
                    self.toolsLoader.load_tools()
            except Exception as e:
                logger.info(f"Error loading tools: {e}. Deleting the tool.")
                yield {
                    "role": "assistant",
                    "content": f"Error loading tools: {e}. Deleting the tool.\n",
                    "metadata": {
                        "title": "Trying to load the newly created tool",
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
                match message.get("role"):
                    case "user":
                        role = "user"
                        if isinstance(message["content"], tuple):
                            path = message["content"][0]
                            try:
                                image_bytes = open(path, "rb").read()
                                mime_type, _ = mimetypes.guess_type(path)
                                parts = [
                                    types.Part.from_bytes(
                                        data=image_bytes,
                                        mime_type=mime_type
                                    ),
                                ]
                            except Exception as e:
                                logger.error(f"Error uploading file: {e}")
                                parts = [types.Part.from_text(
                                    text="Error uploading file: "+str(e))]
                            formatted_history.append(
                                types.Content(
                                    role=role,
                                    parts=parts
                                ))
                            continue
                        else:
                            parts = [types.Part.from_text(
                                text=message.get("content", ""))]
                    case "memories":
                        role = "user"
                        parts = [types.Part.from_text(
                            text="Here are the relevant memories for the user's query: "+message.get("content", ""))]
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
                        parts = [types.Part.from_text(
                            text=message.get("content", ""))]
                formatted_history.append(types.Content(
                    role=role,
                    parts=parts
                ))
        return formatted_history

    def get_k_memories(self, query, k=5, threshold=0.0):
        raw_memories = MemoryManager().get_memories()
        memories = []
        for i in range(len(raw_memories)):
            memories.append(raw_memories[i]['memory'])
        if len(memories) == 0:
            return []
        top_k = min(k, len(memories))
        # Semantic Retrieval with GPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        else:
            device = 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        doc_embeddings = model.encode(
            memories, convert_to_tensor=True, device=device)
        query_embedding = model.encode(
            query, convert_to_tensor=True, device=device)
        similarity_scores = model.similarity(
            query_embedding, doc_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)
        results = []
        for score, idx in zip(scores, indices):
            if score >= threshold:
                results.append(raw_memories[idx.item()])
        return results

    def run(self, messages):
        try:
            if self.check_mode(Mode.ENABLE_MEMORY) and len(messages) > 0:
                memories = self.get_k_memories(
                    messages[-1]['content'], k=5, threshold=0.1)
                if len(memories) > 0:
                    messages.append({
                        "role": "memories",
                        "content": f"{memories}",
                    })
                    messages.append({
                        "role": "assistant",
                        "content": f"Memories: \n```json\n{format_tool_response(memories)}\n```\n",
                        "metadata": {"title": "Memories"}
                    })
                    yield messages
        except Exception as e:
            pass
        yield from self.invoke_manager(messages)

    def invoke_manager(self, messages):
        chat_history = self.format_chat_history(messages)
        logger.debug(f"Chat history: {chat_history}")
        try:
            response_stream = suppress_output(
                self.generate_response)(chat_history)
            full_text = ""  # Accumulate the text from the stream
            function_calls = []
            function_call_requests = []
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    if chunk.text.strip() != "":
                        yield messages + [{
                            "role": "assistant",
                            "content": full_text
                        }]
                    else:
                        print("Empty chunk received")
                        print(chunk)
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        function_call_requests.append({
                            "role": "function_call",
                            "content": repr(candidate.content),
                        })
                        for part in candidate.content.parts:
                            if part.function_call:
                                function_calls.append(part.function_call)
            if full_text.strip() != "":
                messages.append({
                    "role": "assistant",
                    "content": full_text,
                })
            if function_call_requests:
                messages = messages + function_call_requests
            yield messages
        except Exception as e:
            print(messages)
            print(chat_history)
            messages.append({
                "role": "assistant",
                "content": f"Error generating response: {str(e)}",
                "metadata": {"title": "Error generating response"}
            })
            logger.error(f"Error generating response{e}")
            yield messages
            return messages

        # Check if any text was received
        if len(full_text.strip()) == 0 and len(function_calls) == 0:
            messages.append({
                "role": "assistant",
                "content": "No response from the model.",
                "metadata": {"title": "No response from the model."}
            })

        if function_calls and len(function_calls) > 0:
            for call in self.handle_tool_calls(function_calls):
                yield messages + [call]
                if (call.get("role") == "tool"
                        or (call.get("role") == "assistant" and call.get("metadata", {}).get("status") == "done")):
                    messages.append(call)
            yield from self.invoke_manager(messages)
        else:
            yield messages
