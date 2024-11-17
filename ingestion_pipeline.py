import datetime
from pathlib import Path
from typing import List, Optional

from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource
from bytewax import operators as op

from qdrant_client import QdrantClient

from pydantic import TypeAdapter

from pipeline import mocked
from pipeline.embeddings import EmbeddingModelSingleton
from pipeline.models import NewsArticle, Document
from pipeline.qdrant import QdrantVectorOutput

article_adapter = TypeAdapter(List[NewsArticle])

model = EmbeddingModelSingleton(cache_dir=None)

def build_input(is_input_mocked: bool = True,):
    """
    Builds the input source for the dataflow.

    Args:
        is_input_mocked (bool): If True, uses a mocked data source for testing purposes.

    Returns:
        TestingSource: The input source for the dataflow, which is a mocked financial news feed if `is_input_mocked` is True.
    """
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

flow = Dataflow("alpaca_news_input")

alpaca_news_input = op.input("input", flow, build_input())

article_to_class = op.flat_map("class_to_article", alpaca_news_input, lambda messages: article_adapter.validate_python(messages))
_ = op.inspect("articles", article_to_class)
