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
from pipeline.sources.batch import AlpacaNewsBatchInput
from pipeline.sources.stream import AlpacaNewsStreamInput

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
        ingesion_type (IngestionTypes): One of MOCK, BATCH, or STREAM, determines the type of input source to use.
        from_datetime (Optional[datetime.datetime]): If using IngestionTypes.BATCH, the start datetime for the news batch.
        to_datetime (Optional[datetime.datetime]): If using IngestionTypes.BATCH, the end datetime for the news batch.

    Returns:
        TestingSource or AlpacaNewsBatchInput or AlpacaNewsStreamInput: The input source for the dataflow, which is a mocked financial news feed if ingesion_type is IngestionTypes.MOCK,
        an AlpacaNewsBatchInput if IngestionTypes.BATCH, or an AlpacaNewsStreamInput if IngestionTypes.STREAM.
    """
    # if using a mocked input source, return a TestingSource with the mocked financial news
    if ingesion_type == IngestionTypes.MOCK:
        logger.debug("Using mocked input source")
        return TestingSource(mocked.financial_news)

    # if using a batch input source, return an AlpacaNewsBatchInput with the specified from and to datetimes
    elif ingesion_type == IngestionTypes.BATCH:
        logger.debug("Using batch input source")
        assert (
            from_datetime is not None and to_datetime is not None
        ), "from_datetime and to_datetime must be provided when is_batch is True"
        return AlpacaNewsBatchInput(from_datetime=from_datetime, to_datetime=to_datetime)

    # if using a stream input source, return an AlpacaNewsStreamInput with the specified tickers
    elif ingesion_type == IngestionTypes.STREAM:
        logger.debug("Using stream input source")
        return AlpacaNewsStreamInput(tickers=["*"])

    # if ingesion_type is invalid, raise a ValueError
    else:
        raise ValueError(f"Invalid ingestion type: {ingesion_type}")
    
def build_output(model: EmbeddingModelSingleton, in_memory: bool = False):
    """
    Constructs a QdrantVectorOutput object configured for in-memory or cloud storage.

    QdrantVectorOutput is a type of dynamic output that supports at-least-once processing.
    When used with a stream input source, it will deduplicate messages and ensure that
    messages are not lost in the event of a restart.

    :param model: The embedding model that provides the vector size for the output.
    :param in_memory: If True, the output will be stored in memory, which is faster
        but less persistent than storing it in a database. If False, the output will
        be stored in a database, which is slower but more persistent.

    :return: A QdrantVectorOutput object configured for use with Qdrant, either in-memory
        or persistent.
    """
    # If in_memory is True, use an in-memory Qdrant client, which is faster but
    # less persistent than storing the output in a database.
    if in_memory:
        return QdrantVectorOutput(
            # The vector size is determined by the embedding model.
            vector_size=model.max_input_length,
            # The in-memory Qdrant client is specified by providing ":memory:" as the
            # URL.
            client=QdrantClient(":memory:"),
        )
    else:
        # If in_memory is False, use a persistent Qdrant client, which is slower
        # but more persistent than storing the output in memory.
        return QdrantVectorOutput(
            # The vector size is determined by the embedding model.
            vector_size=model.max_input_length,
        )
    
def validate_and_parse_messages(messages):
    """
    Safely validate and parse messages to NewsArticle objects.
    
    Args:
        messages: Raw messages from the input source
        
    Returns:
        List of validated NewsArticle objects
    """
    try:
        # If the input 'messages' is not already a list, encapsulate it in a list
        # This ensures that the subsequent list processing logic can handle both
        # single and multiple message scenarios uniformly.
        if not isinstance(messages, list):
            messages = [messages]
            
        # Initializes a TypeAdapter for the List of NewsArticle.
        # TypeAdapter is used to validate and parse data structures into
        # the specified type, which in this case is a list of NewsArticle.
        article_adapter = TypeAdapter(List[NewsArticle])
        
        # Validate and parse the input messages to a list of NewsArticle objects.
        # This is done using the TypeAdapter, which ensures that each element
        # in the 'messages' conforms to the NewsArticle structure.
        validated_articles = article_adapter.validate_python(messages)
        
        # Log an informational message indicating the number of successfully
        # validated articles. This helps in tracking the progress and success
        # of the validation step.
        logger.info(f"Successfully validated {len(validated_articles)} articles")
        
        # Return the list of validated NewsArticle objects.
        return validated_articles
    
    except Exception as e:
        # If an error occurs during validation or parsing, log an error message
        # with the exception details. This provides insight into what went wrong.
        logger.error(f"Error validating messages: {str(e)}")
        
        # Log a debug message with the problematic messages that caused the error.
        # This can be used for further investigation or debugging.
        logger.debug(f"Problematic messages: {messages}")
        
        # Return an empty list as a fallback to indicate that no valid articles
        # were parsed due to the encountered error.
        return []


def build_dataflow(
        ingestion_type: IngestionTypes = IngestionTypes.MOCK, 
        from_datetime: Optional[datetime.datetime] = None, 
        to_datetime: Optional[datetime.datetime] = None
    ):

    # Obtain a singleton instance of the embedding model.
    # The EmbeddingModelSingleton is used for both chunking and embedding operations.
    model = EmbeddingModelSingleton(cache_dir=None)

    # Create a new Bytewax dataflow named "alpaca_news_input".
    # This dataflow will define the processing steps for ingesting and transforming news data.
    flow = Dataflow("alpaca_news_input")

    # Initialize the Bytewax input operator with the specified ingestion type and date range.
    # This operator will fetch or generate the raw news data for further processing.
    alpaca_news_input = op.input("input", flow, build_input(ingestion_type, from_datetime, to_datetime))
    # _ = op.inspect("inspect_input", alpaca_news_input)

    # Transform each item from the input into a NewsArticle object using validation and parsing.
    # The flat_map operator applies the validation function and flattens the output.
    article_to_class = op.flat_map("class_to_article", alpaca_news_input, validate_and_parse_messages)
    # _ = op.inspect("articles", article_to_class)

    # Convert each NewsArticle object into a Document object.
    # The map operator applies the conversion function to each article.
    document = op.map("document", article_to_class, lambda article: article.to_document())
    # _ = op.inspect("inspect_document", document)

    # Compute chunks for each Document object using the embedding model.
    # The map operator applies the chunk computation function to each document.
    compute_chunks = op.map("chunks", document, lambda document: document.compute_chunks(model))
    # _ = op.inspect("inspect_chunks", compute_chunks)

    # Compute embeddings for each chunk in the Document object using the embedding model.
    # The map operator applies the embedding computation function to each document.
    compute_embeddings = op.map("embeddings", compute_chunks, lambda document: document.compute_embeddings(model)) 
    # _ = op.inspect("inspect_embeddings", compute_embeddings)

    # Output the final computed embeddings to the Qdrant Vector Database.
    # The output operator defines where and how the final result is stored.
    op.output("output", compute_embeddings, build_output(model)) 
    # _ = op.inspect("inspect_output", output)

    # Return the constructed dataflow for execution.
    return flow

def build_batch_dataflow(last_n_days: int = 1):

    """
    Build a Bytewax dataflow to ingest news from the last n days.

    This function will create a Bytewax dataflow to ingest news from the last n days.

    Example:
        $ python -m bytewax.run ingestion_pipeline:build_batch_dataflow

    Args:
        last_n_days (int): Number of days to ingest news from. Defaults to 1.
    """
    
    # Get the current datetime
    to_datetime = datetime.datetime.now()

    # Calculate the datetime n days ago
    from_datetime = to_datetime - datetime.timedelta(days=last_n_days)

    # Log the date range for debugging purposes
    logger.info(
        f"Extracting news from {from_datetime} to {to_datetime} [n_days={last_n_days}]"
    )

    # Build the Bytewax dataflow using the specified ingestion type and date range.
    # The build_dataflow function will return a Bytewax dataflow.
    flow = build_dataflow(
        ingestion_type=IngestionTypes.BATCH,
        from_datetime=from_datetime,
        to_datetime=to_datetime,
    )

    # Return the constructed dataflow for execution.
    return flow

def build_stream_dataflow():
    """
    Build a Bytewax dataflow to ingest news from an Alpaca stream.

    This function will create a Bytewax dataflow to ingest news from an Alpaca stream.
    The dataflow will be built with the ingestion type set to IngestionTypes.STREAM.

    Example:
        $ python -m bytewax.run ingestion_pipeline:build_stream_dataflow

    Returns:
        Dataflow: The constructed Bytewax dataflow.
    """
    # Build the Bytewax dataflow using the specified ingestion type.
    # The build_dataflow function will return a Bytewax dataflow.
    flow = build_dataflow(ingestion_type=IngestionTypes.STREAM)

    # Return the constructed dataflow for execution.
    return flow

def build_mock_dataflow():
    """
    Build a Bytewax dataflow to ingest mock news data for testing purposes.

    This function creates a Bytewax dataflow with the ingestion type set to IngestionTypes.MOCK.
    The dataflow will simulate the ingestion of news data without requiring live or batch data sources.
    
    Example:
        $ python -m bytewax.run ingestion_pipeline:build_mock_dataflow

    Returns:
        Dataflow: The constructed Bytewax dataflow for mock data.
    """
    # Build the Bytewax dataflow using the mock ingestion type.
    # The build_dataflow function will return a Bytewax dataflow.
    flow = build_dataflow(ingestion_type=IngestionTypes.MOCK)

    # Return the constructed dataflow for execution.
    return flow


if __name__ == "__main__":
    print("1. Build Batch Dataflow")
    print("2. Build Stream Dataflow")
    print("3. Build Mock Dataflow")

    choice = input("Enter your choice: ")
    if choice == "1":
        days = int(input("Enter the number of days: "))
        flow = build_batch_dataflow(last_n_days=days)
    elif choice == "2":
        flow = build_stream_dataflow()
    elif choice == "3":
        flow = build_mock_dataflow()
    else:
        print("Invalid choice. Exiting.")
        exit(1)

    run_main(flow)