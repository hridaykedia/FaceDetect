import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "faces")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.40"))
