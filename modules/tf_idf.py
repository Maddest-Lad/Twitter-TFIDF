import asyncio
import pickle
import re
from pathlib import Path
import eel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm

from modules.clip_connector import Connector

# Paths for resource files
RESOURCE_PATH = Path("resources")
DATA_PATH = RESOURCE_PATH / "data"
CORPUS_FILE_PATH = RESOURCE_PATH / "corpus.pkl"
TFIDF_MODEL_FILE_PATH = RESOURCE_PATH / "tfidf_model.pkl"


class TFIDFHandler:
    def __init__(self, rebuild=False):
        self.corpus = None
        self.tfidf_model = None
        self.tfidf_matrix = None
        self.connector = Connector()

        if rebuild or not CORPUS_FILE_PATH.exists() or not TFIDF_MODEL_FILE_PATH.exists():
            asyncio.run(self._build_and_save_model())
        else:
            self.corpus = load_object(CORPUS_FILE_PATH)
            self.tfidf_model = load_object(TFIDF_MODEL_FILE_PATH)
            self.tfidf_matrix = self.tfidf_model.transform(self.corpus)

    async def build_corpus(self) -> list:
        """Build a corpus from the data in DATA_PATH"""
        raw_corpus = []
        for path in tqdm(list(DATA_PATH.glob("*.jpg"))):
            raw_corpus.append(clean_string(await self.connector.interrogate_clip(path)))
        return raw_corpus

    async def _build_and_save_model(self):
        self.corpus = await self.build_corpus()
        self.tfidf_model = build_tfidf_model(self.corpus)
        self.tfidf_matrix = self.tfidf_model.transform(self.corpus)
        save_object(self.corpus, CORPUS_FILE_PATH)
        save_object(self.tfidf_model, TFIDF_MODEL_FILE_PATH)

    async def get_caption_async(self, image_path) -> str:
        new_caption = await self.connector.interrogate_clip(image_path)
        return clean_string(new_caption)

    def get_caption(self, image_path) -> str:
        loop = asyncio.get_event_loop()
        new_caption = loop.run_until_complete(self.get_caption_async(image_path))
        return clean_string(new_caption)

    def score_image(self, image_path: Path) -> int:
        """Score a new image by its similarity to the corpus"""
        loop = asyncio.get_event_loop()
        new_caption = loop.run_until_complete(self.get_caption_async(image_path))
        new_caption_tfidf = self.tfidf_model.transform([new_caption])
        similarity_scores = cosine_similarity(new_caption_tfidf, self.tfidf_matrix)
        max_similarity = np.max(similarity_scores)  # Get the maximum similarity score
        score = int(max_similarity * 100)  # Convert to percentage
        return score


def build_tfidf_model(corpus: list) -> TfidfVectorizer:
    """Build a TF-IDF model from a corpus"""
    print("Building TF-IDF model.")
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    return vectorizer


def save_object(obj, filename: Path) -> None:
    """Save an object to a file using pickle"""
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)


def load_object(filename: Path):
    """Load an object from a file using pickle"""
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def clean_string(text: str) -> str:
    """Cleans a string by lowercasing, removing emojis/symbols/flags, and replacing non-standard characters with spaces"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,_-]+", " ", text)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Remove emojis

    return text


if __name__ == "__main__":
    tfidf_handler = TFIDFHandler(rebuild=False)
