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
from sentence_transformers import SentenceTransformer
import torch
from src.tools.default_tools.memory_manager import MemoryManager
from pathlib import Path

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class GeminiManager:
    def __init__(self, toolsLoader: ToolManager = None,
                 system_prompt_file="./src/models/system4.prompt",
                 gemini_model="gemini-2.5-pro-exp-03-25",
                 local_only=False, allow_tool_creation=True,
                 cloud_only=False, use_economy=True,
                 use_memory=True):
        load_dotenv()
        self.toolsLoader: ToolManager = toolsLoader
        if not toolsLoader:
            self.toolsLoader: ToolManager = ToolManager()

        self.local_only = local_only
        self.allow_tool_creation = allow_tool_creation
        self.cloud_only = cloud_only
        self.use_economy = use_economy
        self.use_memory = use_memory

        self.API_KEY = os.getenv("GEMINI_KEY")
        self.client = genai.Client(api_key=self.API_KEY)
        self.toolsLoader.load_tools()
        self.model_name = gemini_model
        self.memory_manager = MemoryManager() if use_memory else None
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
                                file = self.client.files.upload(file=path)
                                formatted_history.append(file)
                            except Exception as e:
                                logger.error(f"Error uploading file: {e}")
                                formatted_history.append(
                                    types.Part.from_text(text="Error uploading file: "+str(e)))
                            continue
                        else:
                            parts = [types.Part.from_text(text=message.get("content", ""))]
                    case "memories":
                        role = "user"
                        parts = [types.Part.from_text(text="Relevant memories: "+message.get("content", ""))]
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
                        parts = [types.Part.from_text(text=message.get("content", ""))]
                formatted_history.append(types.Content(
                    role=role,
                    parts=parts
                ))
        return formatted_history

    def get_k_memories(self, query, k=5, threshold=0.0):
        if not self.use_memory:
            return []
        memories = MemoryManager().get_memories()
        for i in range(len(memories)):
            memories[i] = memories[i]['memory']
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
        doc_embeddings = model.encode(memories, convert_to_tensor=True, device=device)
        query_embedding = model.encode(query, convert_to_tensor=True, device=device)
        similarity_scores = model.similarity(query_embedding, doc_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=top_k)
        results = []
        for score, idx in zip(scores, indices):
            if score >= threshold:
                results.append(memories[idx])
        return results
    
    def run(self, messages):
        try:
            if self.use_memory:
                memories = self.get_k_memories(messages[-1]['content'], k=5, threshold=0.1)
                if len(memories) > 0:
                    messages.append({
                        "role": "memories",
                        "content": f"{memories}",
                    })
                    messages.append({
                        "role": "assistant",
                        "content": f"Memories: {memories}",
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
            response = suppress_output(self.generate_response)(chat_history)
        except Exception as e:
            messages.append({
                "role": "assistant",
                "content": f"Error generating response: {str(e)}",
                "metadata": {"title": "Error generating response"}
            })
            logger.error(f"Error generating response{e}")
            yield messages
            return messages
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
            yield from self.invoke_manager(messages)
        yield messages
