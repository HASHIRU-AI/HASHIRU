import importlib

__all__ = ['GetWebsiteTool']


class GetWebsiteTool():
    dependencies = ["requests", "beautifulsoup4==4.13.3"]

    inputSchema = {
        "name": "GetWebsiteTool",
        "description": "Returns the content of a website based on a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to fetch content from.",
                },
            },
            "required": ["url"],
        }
    }

    def run(self, **kwargs):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Sec-GPC': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Priority': 'u=0, i',
        }
        print("Running web search")

        url = kwargs.get("url")

        if not url:
            return {
                "status": "error",
                "message": "Missing required parameters: 'url'",
                "output": None
            }
        
        output = None
        requests = importlib.import_module("requests")
        bs4 = importlib.import_module("bs4")
        BeautifulSoup = bs4.BeautifulSoup
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Parse the content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from the parsed HTML
                output = soup.get_text()
            else:
                return {
                    "status": "error",
                    "message": f"Failed to fetch content from {url}. Status code: {response.status_code}",
                    "output": None
                }
            
            # truncate the results to avoid excessive output
            if len(output) > 1000:
                output = output[:1000] + "... (truncated)"

            return {
                "status": "success",
                "message": "Search completed successfully",
                "output": output,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None
            }
