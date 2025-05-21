import importlib
from collections import defaultdict
import re
import time

__all__ = ['GetWebsite']


class GetWebsite():
    dependencies = ["requests", "beautifulsoup4==4.13.3"]

    inputSchema = {
        "name": "GetWebsite",
        "description": "Returns the content of a website with enhanced error handling and output options.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to fetch content from.",
                },
                "output_type": {
                    "type": "string",
                    "enum": ["summary", "full_text"],
                    "description": "The type of output to return. 'summary' returns a summary of the text, 'full_text' returns the full text content.",
                    "default": "full_text"
                },
                "css_selector": {
                    "type": "string",
                    "description": "A CSS selector to extract specific content from the page.",
                }
            },
            "required": ["url"],
        }
    }

    def summarize_text(self, text):
        # Clean the text more thoroughly
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)  # Remove special characters except periods

        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s]

        # Calculate word frequencies
        word_frequencies = defaultdict(int)
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                word_frequencies[word] += 1

        # Normalize word frequencies
        total_words = sum(word_frequencies.values())
        if total_words > 0:
            for word in word_frequencies:
                word_frequencies[word] /= total_words

        # Calculate sentence scores based on word frequencies, sentence length, and coherence
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            for word in words:
                score += word_frequencies[word]

            # Consider sentence length
            sentence_length_factor = 1 - abs(len(words) - 15) / 15  # Prefer sentences around 15 words
            score += sentence_length_factor * 0.1

            # Add a coherence score
            if i > 0 and sentences[i - 1] in sentence_scores:
                previous_sentence_words = sentences[i - 1].lower().split()
                common_words = set(words) & set(previous_sentence_words)
                coherence_score = len(common_words) / len(words)
                score += coherence_score * 0.1

            sentence_scores[sentence] = score

        # Get the top 3 sentences with the highest scores
        ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

        # Generate the summary
        summary = ". ".join(ranked_sentences) + "."
        return summary

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
        print("Running enhanced web scraper")

        url = kwargs.get("url")
        output_type = kwargs.get("output_type", "summary")
        css_selector = kwargs.get("css_selector")

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
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response.encoding = response.apparent_encoding  # Handle encoding

            # Parse the content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            if css_selector:
                # Extract text from the selected elements
                elements = soup.select(css_selector)
                text = '\n'.join([element.get_text() for element in elements])
            else:
                # Extract text from the parsed HTML
                text = soup.get_text()

            if output_type == "summary":
                # Summarize the text
                output = self.summarize_text(text)
            elif output_type == "full_text":
                output = text
            else:
                return {
                    "status": "error",
                    "message": f"Invalid output_type: {output_type}",
                    "output": None
                }


            return {
                "status": "success",
                "message": "Search completed successfully",
                "output": output,
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "output": None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "output": None
            }
