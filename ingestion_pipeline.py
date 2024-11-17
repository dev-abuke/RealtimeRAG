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
    """
    Constructs a QdrantVectorOutput object configured for in-memory or cloud storage.

    Args:
        model (EmbeddingModelSingleton): The embedding model providing the vector size.
        in_memory (bool): If True, configures the output to use an in-memory Qdrant client.

    Returns:
        QdrantVectorOutput: An output object configured for use with Qdrant, either in-memory or persistent.
    """
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

# initialize bytewax flow with Mock Input for Now
alpaca_news_input = op.input("input", flow, build_input())

# convert each of the out output from alpaca_news_input to apydantic NewsArticle model
article_to_class = op.flat_map("class_to_article", alpaca_news_input, lambda messages: article_adapter.validate_python(messages))
_ = op.inspect("articles", article_to_class)

# convert each of the out output from article_to_class to apydantic Document model
document = op.map("document", article_to_class, lambda article: article.to_document())
_ = op.inspect("inspect_document", document)

# compute chunks from the documents
compute_chunks = op.map("chunks", document, lambda document: document.compute_chunks(model))
_ = op.inspect("inspect_chunks", compute_chunks)

# compute embeddings for each chunks
compute_embeddings = op.map("embeddings", compute_chunks, lambda document: document.compute_embeddings(model)) 
_ = op.inspect("inspect_embeddings", compute_embeddings)

# Sink the output to Qdrant Vector DB
output = op.output("output", compute_embeddings, build_output(model)) 