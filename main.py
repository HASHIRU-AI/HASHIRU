from typing import List
from googlesearch import search
from CEO.CEO import OllamaModelManager
from CEO.tool_loader import ToolLoader

# Define the web search tool function.
def web_search(website: str, query: str) -> List[str]:
    """
    Searches the specified website for the given query.
    The search query is formed by combining the website domain and the query string.
    """
    search_query = f"site:{website} {query}"
    results = []
    for result in search(search_query, num_results=10):
        # Filter out irrelevant search pages
        if "/search?num=" not in result:
            results.append(result)
    return results

if __name__ == "__main__":
    # Define the tool metadata for orchestration.
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'web_search',
                'description': 'Search for results on a specified website using a query string. '
                               'The CEO model should define which website to search from and the query to use.',
                'parameters': {
                    'type': 'object',
                    'required': ['website', 'query'],
                    'properties': {
                        'website': {'type': 'string', 'description': 'The website domain to search from (e.g., huggingface.co)'},
                        'query': {'type': 'string', 'description': 'The search query to use on the specified website'},
                    },
                },
            },
        }
    ]

    # Load the tools using the ToolLoader class.
    tool_loader = ToolLoader()
    # tools.extend(tool_loader.getTools())

    # Create the Ollama model manager and ensure the model is set up.
    model_manager = OllamaModelManager(toolsLoader=tool_loader)
    
    # Example prompt instructing the CEO model to create a strategy for Ashton Hall.
    # The prompt explicitly mentions that it can use the web_search tool if needed,
    # and that it is allowed to choose the website for the search.
    task_prompt = (
        "Should I wear a sweater today?"
    )
    
    # Request a CEO response with the prompt.
    response = model_manager.request([{"role": "user", "content": task_prompt}])
    print("\nCEO Response:", response)
