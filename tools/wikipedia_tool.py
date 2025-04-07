import importlib

__all__ = ['WikipediaTool']


class WikipediaTool():
    dependencies = ["requests==2.32.3", "beautifulsoup4==4.13.3"]

    inputSchema = {
        "name": "WikipediaTool",
        "description": "Searches Wikipedia for a given question and returns a short summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The topic or question to search on Wikipedia.",
                }
            },
            "required": ["question"],
        }
    }

    def run(self, **kwargs):
        question = kwargs.get("question")
        if not question:
            return {
                "status": "error",
                "message": "Missing required parameter: 'question'",
                "output": None
            }

        print(f"Searching Wikipedia for: {question}")

        requests = importlib.import_module("requests")
        bs4 = importlib.import_module("bs4")
        BeautifulSoup = bs4.BeautifulSoup

        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": question,
            "srlimit": 1,
        }

        try:
            response = requests.get(search_url, params=search_params)
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": "Wikipedia API request failed.",
                    "output": None,
                }

            data = response.json()
            search_results = data.get("query", {}).get("search", [])

            if not search_results:
                return {
                    "status": "error",
                    "message": "No results found on Wikipedia.",
                    "output": None,
                }

            top_result = search_results[0]["title"]
            page_url = f"https://en.wikipedia.org/wiki/{top_result.replace(' ', '_')}"
            print(f"Fetching full content from: {page_url}")

            html_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{top_result.replace(' ', '_')}"
            html_response = requests.get(html_url)

            if html_response.status_code != 200:
                return {
                    "status": "error",
                    "message": "Failed to fetch article content.",
                    "output": None,
                }

            soup = BeautifulSoup(html_response.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p") if p.get_text()]
            full_text = " ".join(paragraphs)
            summary = " ".join(full_text.split(". ")[:5])  # First 5 sentences

            output_text = f"**{top_result}**\n{summary}...\n[Read more]({page_url})"

            return {
                "status": "success",
                "message": "Wikipedia article summary retrieved successfully.",
                "output": output_text,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred: {str(e)}",
                "output": None,
            }
