import os
import sys
import logging
import json
from typing import List

import chromadb
from sklearn.manifold import TSNE
import numpy as np

from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity


# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange",
          "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        chroma_client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = chroma_client.get_or_create_collection("products")
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            return [Opportunity(**item) for item in data]
        return []

    def write_memory(self) -> None:
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        if not os.path.exists(cls.MEMORY_FILENAME):
            return
        with open(cls.MEMORY_FILENAME, "r") as file:
            data = json.load(file)
        with open(cls.MEMORY_FILENAME, "w") as file:
            json.dump(data[:2], file, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        chroma_client = chromadb.PersistentClient(path=cls.DB)
        collection = chroma_client.get_or_create_collection("products")

        result = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=max_datapoints,
        )

        embs = result.get("embeddings")
        if embs is None:
            return [], np.empty((0, 3)), []

        vectors = np.asarray(embs, dtype=float)
        if vectors.size == 0:
            return [], np.empty((0, 3)), []

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        categories = [(m.get("category") if isinstance(m, dict) else None)
                      for m in metadatas]
        colors = [COLORS[CATEGORIES.index(
            c)] if c in CATEGORIES else "gray" for c in categories]

        n_samples = vectors.shape[0]
        if n_samples < 2:
            return documents, np.empty((0, 3)), colors

        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=3, random_state=42,
                    perplexity=perplexity, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)

        return documents, reduced_vectors, colors


if __name__ == "__main__":
    # Optional: only load .env on execution, not on import
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    DealAgentFramework().run()
