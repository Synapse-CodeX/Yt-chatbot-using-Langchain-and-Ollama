import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"

os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY