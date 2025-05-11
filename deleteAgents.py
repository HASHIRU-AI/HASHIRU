import json
import ollama
from src.manager.utils.streamlit_interface import output_assistant_response

with open("./src/models/models.json", "r", encoding="utf8") as f:
    models = f.read()
models = json.loads(models)
for agent in models:
    output_assistant_response(f"Deleting agent: {agent}")
    try:
        ollama.delete(agent)
    except Exception as e:
        output_assistant_response(f"Error deleting agent {agent}: {e}")
    with open("./models/models.json", "w", encoding="utf8") as f:
        f.write(json.dumps({}, indent=4))