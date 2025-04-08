from google.genai import types
from typing import List
from googlesearch import search
from CEO.CEO import GeminiManager
from CEO.tool_loader import ToolLoader
import warnings

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

    model_manager = GeminiManager(toolsLoader=tool_loader, gemini_model="gemini-2.0-flash")
    
    # Example prompt instructing the CEO model to create a strategy for Ashton Hall.
    # The prompt explicitly mentions that it can use the web_search tool if needed,
    # and that it is allowed to choose the website for the search.
    task_prompt = (
        "Should I invest in TSLA stocks? "
    )
    
    # Request a CEO response with the prompt.
    user_prompt_content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=task_prompt)],
    )
    response = model_manager.request([user_prompt_content])
    print("\nCEO Response:", response)
