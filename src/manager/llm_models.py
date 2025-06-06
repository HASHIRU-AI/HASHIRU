from abc import ABC, abstractmethod
import ollama
from pydantic import BaseModel
from pathlib import Path
from google import genai
from google.genai import types
from mistralai import Mistral
from groq import Groq
from src.manager.utils.streamlit_interface import output_assistant_response


class AbstractModelManager(ABC):
    def __init__(self, model_name, system_prompt_file="system.prompt"):
        self.model_name = model_name
        script_dir = Path(__file__).parent
        self.system_prompt_file = script_dir / system_prompt_file

    @abstractmethod
    def is_model_loaded(self, model):
        pass

    @abstractmethod
    def create_model(self, base_model, context_window=4096, temperature=0):
        pass

    @abstractmethod
    def request(self, prompt):
        pass

    @abstractmethod
    def delete(self):
        pass

class OllamaModelManager(AbstractModelManager):
    def is_model_loaded(self, model):
        loaded_models = [m.model for m in ollama.list().models]
        return model in loaded_models or f'{model}:latest' in loaded_models

    def create_model(self, base_model, context_window=4096, temperature=0):
        with open(self.system_prompt_file, 'r') as f:
            system = f.read()
        
        if not self.is_model_loaded(self.model_name):
            output_assistant_response(f"Creating model {self.model_name}")
            ollama.create(
                model=self.model_name,
                from_=base_model,
                system=system,
                parameters={
                    "num_ctx": context_window,
                    "temperature": temperature
                }
            )

    def request(self, prompt):
        response = ollama.chat(
            model=self.model_name, 
            messages=[{"role": "user", "content": prompt}],
        )
        response = response['message']['content']
        return response
    
    def delete(self):
        if self.is_model_loaded("C2Rust:latest"):
            output_assistant_response(f"Deleting model {self.model_name}")
            ollama.delete("C2Rust:latest")
        else:
            output_assistant_response(f"Model {self.model_name} not found, skipping deletion.")

class GeminiModelManager(AbstractModelManager):
    def __init__(self, api_key):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"
        # read system prompt from file
        with open(self.system_prompt_file, 'r') as f:
            self.system_instruction = f.read()


    def is_model_loaded(self, model):
        # Check if the specified model is the one set in the manager
        return model == self.model

    def create_model(self, base_model=None, context_window=4096, temperature=0):
        # Initialize the Gemini model settings (if applicable)
        self.model = base_model if base_model else "gemini-2.0-flash"

    def request(self, prompt, temperature=0, context_window=4096):
        # Request response from the Gemini model
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=context_window,
                system_instruction=self.system_instruction,
            )
        )
        return response.text

    def delete(self):
        # Implement model deletion logic (if applicable)
        self.model = None

class MistralModelManager(AbstractModelManager):
    def __init__(self, api_key, model_name="mistral-small-latest", system_prompt_file="system.prompt"):
        super().__init__()
        self.client = Mistral(api_key=api_key)
        self.model = model_name
        # read system prompt from file
        with open(self.system_prompt_file, 'r') as f:
            self.system_instruction = f.read()

    def is_model_loaded(self, model):
        # Check if the specified model is the one set in the manager
        return model == self.model

    def create_model(self, base_model=None, context_window=4096, temperature=0):
        # Initialize the Mistral model settings (if applicable)
        self.model = base_model if base_model else "mistral-small-latest"

    def request(self, prompt, temperature=0, context_window=4096):
        # Request response from the Mistral model
        response = self.client.chat.complete(
            messages=[
            {
                "role":"user",
                "content": self.system_instruction + "\n" + prompt,
            }
            ],
            model=self.model,
            temperature=temperature,
            max_tokens=context_window,
        )
        return response.text

    def delete(self):
        # Implement model deletion logic (if applicable)
        self.model = None

class GroqModelManager(AbstractModelManager):
    def __init__(self, api_key, model_name="llama-3.3-70b-versatile", system_prompt_file="system.prompt"):
        super().__init__(model_name, system_prompt_file)
        self.client = Groq(api_key=api_key)

    def is_model_loaded(self, model):
        # Groq models are referenced by name; assume always available if name matches
        return model == self.model_name

    def create_model(self, base_model=None, context_window=4096, temperature=0):
        # Groq does not require explicit creation; no-op
        if not self.is_model_loaded(self.model_name):
            output_assistant_response(f"Model {self.model_name} is not available on Groq.")

    def request(self, prompt, temperature=0, context_window=4096):
        # Read system instruction
        with open(self.system_prompt_file, 'r') as f:
            system_instruction = f.read()

        # Build messages
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

        # Send request
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=temperature
        )

        # Extract and return content
        return response.choices[0].message.content

    def delete(self):
        # No deletion support for Groq-managed models
        output_assistant_response(f"Deletion not supported for Groq model {self.model_name}.")
