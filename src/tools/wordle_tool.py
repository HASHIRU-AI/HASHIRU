
import re
import random
import json
import requests

__all__ = ['WordleTool']

class WordleTool():
    dependencies = []

    inputSchema = {
        "name": "WordleTool",
        "description": "A tool to play Wordle.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["new_game", "guess", "reset"],
                    "description": "The action to perform: new_game, guess, or reset."
                },
                "guess": {
                    "type": "string",
                    "description": "A 5-letter word guess. Required for 'guess' action."
                }
            },
            "required": ["action"],
        },
        "invoke_cost": 0.2,
    }

    def __init__(self):
        self.secret_word = None
        self.word_list_url = "https://github.com/kiprobinson/wordle-solver/raw/main/app/resources/word-list.txt"
        self.file_path = "src/data/secret_word.json"  # Path for storing the secret word

    def _load_word_list(self):
        try:
            response = requests.get(self.word_list_url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            word_list_str = response.text
            word_list = word_list_str.strip().split("\n")
            return word_list
        except requests.exceptions.RequestException as e:
            print(f"Error fetching word list: {e}")
            return ['crane', 'house', 'table', 'chair', 'apple']

    def _get_secret_word(self):
        word_list = self._load_word_list()
        return random.choice(word_list)

    def _store_secret_word(self, word):
        with open(self.file_path, "w") as f:
            json.dump({"secret_word": word}, f)
        return {"result": "Secret word stored successfully."}

    def _retrieve_secret_word(self):
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                return {"result": data["secret_word"]}
        except FileNotFoundError:
            return {"result": None}

    def run(self, **kwargs):
        action = kwargs.get("action")
        guess = kwargs.get("guess")

        if action == "new_game":
            self.secret_word = self._get_secret_word()
            self._store_secret_word(self.secret_word)
            return "New word generated. Please make your guess."

        elif action == "reset":
            self.secret_word = None
            return "Game reset."

        elif action == "guess":
            # Retrieve secret word from file
            secret_word_data = self._retrieve_secret_word()
            if secret_word_data["result"] is None:
                return "No secret word found. Please start a new game."
            self.secret_word = secret_word_data["result"]

            if not guess:
                return "Please provide a guess."

            guess = guess.lower()
            if not re.match("^[a-z]{5}$", guess):
                return "Invalid input. Please enter a 5-letter word."

            result = ""
            for i in range(5):
                if guess[i] == self.secret_word[i]:
                    result += "G"  # Green: Correct letter, correct position
                elif guess[i] in self.secret_word:
                    result += "Y"  # Yellow: Correct letter, wrong position
                else:
                    result += "X"  # Gray: Incorrect letter
            return result
        else:
            return "Invalid action. Please choose 'new_game', 'guess', or 'reset'."
