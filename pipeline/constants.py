EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384
EMBEDDING_MODEL_DEVICE = "cpu"

CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Qdrant Cloud
QDRANT_URL: str = "https://73bdd42b-86a7-49fc-bcf4-e6bf85cfca17.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY: str = 'pG-09pBVQTDpHkLVL3b3m_A_nEZLGYg88ew8_wFb5BtkasvGpyHOlQ'
VECTOR_DB_OUTPUT_COLLECTION_NAME = "alpaca_financial_news"

# OpenAI
OPENAI_API_KEY: str = "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"
LLM_MODEL_NAME: str = "gpt-4o-mini"
LLM_BASE_URL: str = "https://openrouter.ai/api/v1"

# Alpaca News
ALPACA_NEWS_API_KEY: str = "PKM19APHZSD7EDUI20D6"
APACA_NEWS_SECRET: str = "GifphcRRfVCyc4VTfaTBg9z4MZT5nP3rdZVgkq0x"