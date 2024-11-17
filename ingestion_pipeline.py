import datetime
from pathlib import Path
from typing import List, Optional

from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource

from qdrant_client import QdrantClient

from ingestion_pipeline import mocked
from pipeline.embeddings import EmbeddingModelSingleton
from pipeline.models import NewsArticle, Document
from pipeline.qdrant import QdrantVectorOutput

from bytewax import operators as op

model = EmbeddingModelSingleton(cache_dir=None)

def build_input(is_input_mocked: bool = True,):
    if is_input_mocked:
        return TestingSource(mocked.financial_news)