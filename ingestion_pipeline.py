# system imports
import datetime
from enum import Enum
from loguru import logger
from typing import List, Optional

# bytewax import
from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource
from bytewax import operators as op
from bytewax.testing import run_main

# qdrant imports
from qdrant_client import QdrantClient

# pydantic imports
from pydantic import TypeAdapter

# local imports
from pipeline import mocked
from pipeline.embeddings import EmbeddingModelSingleton
from pipeline.models import NewsArticle, Document
from pipeline.qdrant import QdrantVectorOutput

article_adapter = TypeAdapter(List[NewsArticle])

class IngestionTypes(Enum):
    STREAM = "stream"
    BATCH = "batch"
    MOCK = "mock"

def build_input(
        ingesion_type: IngestionTypes = IngestionTypes.MOCK,
        from_datetime: Optional[datetime.datetime] = None, 
        to_datetime: Optional[datetime.datetime] = None
    ):
    """
    Builds the input source for the dataflow.

    Args:
        is_input_mocked (bool): If True, uses a mocked data source for testing purposes.

    Returns:
        TestingSource: The input source for the dataflow, which is a mocked financial news feed if `is_input_mocked` is True.
    """
    if ingesion_type == IngestionTypes.MOCK:
        return TestingSource(mocked.financial_news)
    
    elif ingesion_type == IngestionTypes.BATCH:
         return AlpacaNewsBatchInput(from_datetime=from_datetime, to_datetime=to_datetime, tickers=["*"])

    elif ingesion_type == IngestionTypes.STREAM:
        return AlpacaNewsStreamInput(tickers=["*"])

    else:
        raise ValueError(f"Invalid ingestion type: {ingesion_type}")
    
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

def build_dataflow(
        ingestion_type: IngestionTypes = IngestionTypes.MOCK, 
        from_datetime: Optional[datetime.datetime] = None, 
        to_datetime: Optional[datetime.datetime] = None
):

    model = EmbeddingModelSingleton(cache_dir=None)

    flow = Dataflow("alpaca_news_input")

    # initialize bytewax flow with Mock Input for Now
    alpaca_news_input = op.input("input", flow, build_input(ingestion_type, from_datetime, to_datetime))
    # _ = op.inspect("inspect_input", alpaca_news_input)

    # convert each of the out output from alpaca_news_input to apydantic NewsArticle model
    article_to_class = op.flat_map("class_to_article", alpaca_news_input, lambda messages: article_adapter.validate_python(messages))
    # _ = op.inspect("articles", article_to_class)

    # convert each of the out output from article_to_class to apydantic Document model
    document = op.map("document", article_to_class, lambda article: article.to_document())
    # _ = op.inspect("inspect_document", document)

    # compute chunks from the documents
    compute_chunks = op.map("chunks", document, lambda document: document.compute_chunks(model))
    # _ = op.inspect("inspect_chunks", compute_chunks)

    # compute embeddings for each chunks
    compute_embeddings = op.map("embeddings", compute_chunks, lambda document: document.compute_embeddings(model)) 
    # _ = op.inspect("inspect_embeddings", compute_embeddings)

    # Sink the output to Qdrant Vector DB
    op.output("output", compute_embeddings, build_output(model)) 
    # _ = op.inspect("inspect_output", output)

    return flow

def build_batch_dataflow(last_n_days: int = 1):

    to_datetime = datetime.datetime.now()
    from_datetime = to_datetime - datetime.timedelta(days=last_n_days)

    logger.info(
        f"Extracting news from {from_datetime} to {to_datetime} [n_days={last_n_days}]"
    )

    flow = build_dataflow(
        ingestion_type=IngestionTypes.BATCH,
        from_datetime=from_datetime,
        to_datetime=to_datetime,
    )

    return flow


if __name__ == "__main__":
    run_main(build_dataflow())