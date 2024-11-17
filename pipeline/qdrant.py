import os
from typing import Optional

from bytewax.outputs import DynamicSink, StatelessSinkPartition
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from pipeline import constants
from pipeline.models import Document
from pipeline.base import SingletonMeta

class QdrantVectorOutput(DynamicSink):
    """A class representing a Qdrant vector output.

    This class is used to create a Qdrant vector output, which is a type of dynamic output that supports
    at-least-once processing. Messages from the resume epoch will be duplicated right after resume.

    Args:
        vector_size (int): The size of the vector.
        collection_name (str, optional): The name of the collection.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
        client (Optional[QdrantClient], optional): The Qdrant client. Defaults to None.
    """

    def __init__(
        self,
        vector_size: int,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,
    ):
        self._collection_name = collection_name
        self._vector_size = vector_size

        if client:
            self.client = client
        else:
            self.client = build_qdrant_client()

        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def build(self, step_id, worker_index, worker_count):
        """Builds a QdrantVectorSink object.

        Args:
            worker_index (int): The index of the worker.
            worker_count (int): The total number of workers.

        Returns:
            QdrantVectorSink: A QdrantVectorSink object.
        """
        print(f"Building QdrantVectorSink for worker: step_id {step_id} worker_index {worker_index} and worker_count {worker_count}")

        return QdrantVectorSink(self.client, self._collection_name)


def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Builds a QdrantClient object with the given URL and API key.

    Args:
        url (Optional[str]): The URL of the Qdrant server. If not provided,
            it will be read from the QDRANT_URL environment variable.
        api_key (Optional[str]): The API key to use for authentication. If not provided,
            it will be read from the QDRANT_API_KEY environment variable.

    Raises:
        KeyError: If the QDRANT_URL or QDRANT_API_KEY environment variables are not set
            and no values are provided as arguments.

    Returns:
        QdrantClient: A QdrantClient object connected to the specified Qdrant server.
    """

    if url is None:
        try:
            # TODO: use environment variable
            url = 'https://73bdd42b-86a7-49fc-bcf4-e6bf85cfca17.us-east4-0.gcp.cloud.qdrant.io:6333' # os.environ["QDRANT_URL"]
        except KeyError:
            raise KeyError(
                "QDRANT_URL must be set as environment variable or manually passed as an argument."
            )

    if api_key is None:
        try:
            # TODO: use environment variable
            api_key = 'pG-09pBVQTDpHkLVL3b3m_A_nEZLGYg88ew8_wFb5BtkasvGpyHOlQ' # os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
            )

    client_singleton = QdrantClientSingleton(
        url=url,
        api_key=api_key
    )
    
    return client_singleton.client


class QdrantVectorSink(StatelessSinkPartition):
    """
    A sink that writes document embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client to use for writing.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        self._client = client
        self._collection_name = collection_name

    def write_batch(self, document: list[Document]):
        print(f"Writing {len(document)} embeddings to Qdrant...")
        ids, payloads = document[0].to_payloads()
        print(f"The Payloads in Qdrant &&&######### : {payloads}")
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document[0].embeddings, payloads)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)

class QdrantClientSingleton(metaclass=SingletonMeta):
    """
    Singleton wrapper for QdrantClient.
    Ensures only one instance of QdrantClient is created and reused.
    """
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False
    ):
        """
        Initialize the Qdrant client singleton.
        
        Args:
            url: Optional Qdrant server URL
            api_key: Optional Qdrant API key
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        self._client: Optional[QdrantClient] = None
        self._url = url
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Qdrant client if not already initialized."""
        if self._client is None:
            if self._url is None:
                # Use in-memory storage if no URL provided
                self._client = QdrantClient(":memory:")
            else:
                # Initialize with provided configuration
                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                    prefer_grpc=self._prefer_grpc
                )
    
    @property
    def client(self) -> QdrantClient:
        """
        Get the Qdrant client instance.
        
        Returns:
            QdrantClient: The initialized Qdrant client
        """
        if self._client is None:
            self._initialize_client()
        return self._client