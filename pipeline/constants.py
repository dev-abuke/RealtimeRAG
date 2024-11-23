import os

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_MAX_INPUT_LENGTH = os.getenv("EMBEDDING_MODEL_MAX_INPUT_LENGTH", 384)
EMBEDDING_MODEL_DEVICE = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")

CROSS_ENCODER_MODEL_ID: str = os.getenv("CROSS_ENCODER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Qdrant Cloud
QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
VECTOR_DB_OUTPUT_COLLECTION_NAME = os.getenv("VECTOR_DB_OUTPUT_COLLECTION_NAME", "alpaca_financial_news")  

# OpenAI
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

# Alpaca News
ALPACA_NEWS_API_KEY: str = os.environ["ALPACA_NEWS_API_KEY"]
APACA_NEWS_SECRET: str = os.environ["ALPACA_NEWS_SECRET"]