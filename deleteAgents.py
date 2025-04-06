import json
import ollama

with open("./models/models.json", "r", encoding="utf8") as f:
    models = f.read()
models = json.loads(models)
for agent in models:
    print(f"Deleting agent: {agent}")
    ollama.delete(agent)
    with open("./models/models.json", "w", encoding="utf8") as f:
        f.write(json.dumps({}, indent=4))