import datetime
from pathlib import Path
from typing import List, Optional

from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource

from qdrant_client import QdrantClient

from pipeline import mocked
from pipeline.embeddings import EmbeddingModelSingleton
from pipeline.models import NewsArticle, Document
from pipeline.qdrant import QdrantVectorOutput

from bytewax import operators as op

model = EmbeddingModelSingleton(cache_dir=None)

def build_input(is_input_mocked: bool = True,):
    if is_input_mocked:
        return TestingSource(mocked.financial_news)
    
def build_output(model: EmbeddingModelSingleton, in_memory: bool = False):
    if in_memory:
        return QdrantVectorOutput(
            vector_size=model.max_input_length,
            client=QdrantClient(":memory:"),
        )
    else:
        return QdrantVectorOutput(
            vector_size=model.max_input_length,
        )
    
    