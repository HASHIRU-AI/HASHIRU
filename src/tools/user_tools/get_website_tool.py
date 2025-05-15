import importlib
from collections import defaultdict
import re
import time

__all__ = ['GetWebsiteTool']


class GetWebsiteTool():
    dependencies = ["requests", "beautifulsoup4==4.13.3"]

    inputSchema = {
        "name": "GetWebsiteTool",
        "description": "Returns a summary of the content of a website based on a query string.",
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

    def summarize_text(self, text):
        # Clean the text more thoroughly
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9.\s]', '', text) # Remove special characters except periods

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
            if i > 0 and sentences[i-1] in sentence_scores:
                previous_sentence_words = sentences[i-1].lower().split()
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
                text = soup.get_text()

                # Summarize the text
                output = self.summarize_text(text)
            else:
                return {
                    "status": "error",
                    "message": f"Failed to fetch content from {url}. Status code: {response.status_code}",
                    "output": None
                }

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
