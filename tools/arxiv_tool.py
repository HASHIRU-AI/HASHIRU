import importlib

__all__ = ['ArxivTool']


class ArxivTool():
    dependencies = ["arxiv==2.1.3"]

    inputSchema = {
        "name": "ArxivTool",
        "description": "Searches arXiv for academic papers based on a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for papers (e.g., 'superconductors gem5').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to retrieve. Default is 5.",
                    "default": 5
                }
            },
            "required": ["query"],
        }
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return {
                "status": "error",
                "message": "Missing required parameter: 'query'",
                "output": None
            }

        try:
            arxiv = importlib.import_module("arxiv")
            client = arxiv.Client()

            search = arxiv.Search(
                query=query,
                max_results=max_results,
            )

            papers = []
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.isoformat(),
                    "summary": result.summary.strip(),
                    "pdf_url": result.pdf_url,
                })

            return {
                "status": "success",
                "message": f"Found {len(papers)} paper(s) on arXiv",
                "output": papers,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"arXiv search failed: {str(e)}",
                "output": None,
            }
