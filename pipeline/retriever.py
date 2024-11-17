from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Set
from qdrant_client.http import models
from qdrant_client import QdrantClient
from loguru import logger
from pipeline import constants
from pipeline.embeddings import EmbeddingModelSingleton, CrossEncoderModelSingleton
from pipeline.models import NewsArticle, EmbeddedChunkedArticle
from pipeline.qdrant import build_qdrant_client

@dataclass
class SearchResult:
    """Data class for search results."""
    articles: List[EmbeddedChunkedArticle]
    query: Optional[str] = None
    embedded_queries: Optional[List[List[float]]] = None

class QueryProcessor:
    """Handles query processing and embedding."""
    
    def __init__(self, embedding_model: EmbeddingModelSingleton):
        self.embedding_model = embedding_model
    
    def process_query(self, query: str) -> List[List[float]]:
        """
        Process and embed a query.
        
        Args:
            query: Raw query string
            
        Returns:
            List of embedded query chunks
        """
        cleaned_query = NewsArticle.clean_all(query)
        chunks = NewsArticle.compute_chunks(cleaned_query, self.embedding_model)
        return [self.embedding_model(chunk, to_list=True) for chunk in chunks]

class SearchRequestBuilder:
    """Builds search requests for Qdrant."""
    
    @staticmethod
    def build_requests(
        embedded_queries: List[List[float]],
        limit: int,
        include_payload: bool = True,
        include_vector: bool = True
    ) -> List[models.SearchRequest]:
        """
        Build search requests for each embedded query.
        
        Args:
            embedded_queries: List of query embeddings
            limit: Number of results to return
            include_payload: Whether to include payload in results
            include_vector: Whether to include vectors in results
            
        Returns:
            List of SearchRequest objects
        """
        return [
            models.SearchRequest(
                vector=embedding,
                limit=limit,
                with_payload=include_payload,
                with_vector=include_vector
            )
            for embedding in embedded_queries
        ]

class ArticleProcessor:
    """Processes and ranks retrieved articles."""
    
    def __init__(self, cross_encoder_model: Optional[CrossEncoderModelSingleton] = None):
        self.cross_encoder_model = cross_encoder_model
    
    def process_retrieved_points(
        self,
        retrieved_points: List[List[models.ScoredPoint]],
    ) -> Set[EmbeddedChunkedArticle]:
        """
        Process retrieved points into articles.
        
        Args:
            retrieved_points: List of lists of scored points
            
        Returns:
            Set of unique EmbeddedChunkedArticle objects
        """
        articles = set()
        for chunk_points in retrieved_points:
            logger.debug(f"Processing {len(chunk_points)} retrieved points")
            chunk_articles = {
                EmbeddedChunkedArticle.from_retrieved_point(point)
                for point in chunk_points
            }
            articles.update(chunk_articles)
        return articles
    
    def rank_articles(
        self,
        articles: List[EmbeddedChunkedArticle],
        query: str,
        limit: int
    ) -> List[EmbeddedChunkedArticle]:
        """
        Rank articles using cross-encoder or score sorting.
        
        Args:
            articles: List of articles to rank
            query: Original query string
            limit: Number of articles to return
            
        Returns:
            Ranked and limited list of articles
        """
        if self.cross_encoder_model:
            # TODO: Implement reranking
            # articles = self.rank_articles(query, articles)
            pass
        else:
            articles.sort(key=lambda x: x.score, reverse=True)
        
        return articles[:limit]

class VectorDBRetriever:
    """Main retriever class for vector database search."""
    
    def __init__(
        self,
        cross_encoder_model: Optional[CrossEncoderModelSingleton] = CrossEncoderModelSingleton(),
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        """
        Initialize the retriever.
        
        Args:
            cross_encoder_model: Optional cross encoder for reranking
            collection_name: Name of the vector database collection
        """
        self._query_processor = QueryProcessor(EmbeddingModelSingleton())
        self._article_processor = ArticleProcessor(cross_encoder_model)
        self._vector_db_client: QdrantClient = build_qdrant_client()
        self._collection_name = collection_name
    
    def search(
        self,
        query: str,
        limit: int = 3,
        return_all: bool = False
    ) -> Union[List[EmbeddedChunkedArticle], SearchResult]:
        """
        Search for articles matching the query.
        
        Args:
            query: Search query string
            limit: Number of results to return
            return_all: Whether to return additional search metadata
            
        Returns:
            List of articles or SearchResult object with metadata
        """
        # Process and embed query
        embedded_queries = self._query_processor.process_query(query)
        
        # Adjust limit for reranking if needed
        search_limit = limit * 10 if self._article_processor.cross_encoder_model else limit
        
        # Build and execute search requests
        search_requests = SearchRequestBuilder.build_requests(
            embedded_queries=embedded_queries,
            limit=search_limit
        )
        
        retrieved_points = self._vector_db_client.search_batch(
            collection_name=self._collection_name,
            requests=search_requests,
        )
        logger.info(f"Retrieved {len(retrieved_points)} batches of points")
        
        # Process retrieved points into articles
        articles = self._article_processor.process_retrieved_points(retrieved_points)
        
        # Rank and limit articles
        ranked_articles = self._article_processor.rank_articles(
            articles=list(articles),
            query=query,
            limit=limit
        )
        
        if return_all:
            return SearchResult(
                articles=ranked_articles,
                query=query,
                embedded_queries=embedded_queries
            )
        
        return ranked_articles

if __name__ == "__main__":
    # Initialize retriever
    retriever = VectorDBRetriever()
    
    # Simple search
    results = retriever.search("What happened with AAPL stock?")
    
    # Search with metadata
    detailed_results = retriever.search(
        "Latest news about Tesla",
        limit=5,
        return_all=True
    )