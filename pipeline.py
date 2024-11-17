import datetime
from pathlib import Path
from typing import List, Optional

from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource
from pydantic import parse_obj_as

from qdrant_client import QdrantClient

from pipeline import mocked
# from streaming_pipeline.alpaca_batch import AlpacaNewsBatchInput
# from streaming_pipeline.alpaca_stream import AlpacaNewsStreamInput
from streaming_pipeline.embeddings import EmbeddingModelSingleton
from streaming_pipeline.models import NewsArticle, Document
from streaming_pipeline.qdrant import QdrantVectorOutput

from bytewax import operators as op

model = EmbeddingModelSingleton(cache_dir=None)